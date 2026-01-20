import os
import torch
import json
import pickle
from utils.function import create_logger
from datetime import datetime
from copy import deepcopy
from json_repair import repair_json
import multiprocessing as mp
import re
import argparse
import math
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from utils.inference_entrypoint_ray import (
    llm_load_and_inference_ray,
    mllm_load_and_inference_ray,
    mllm_inference_for_caption_ray,
    clip_extact_feature_ray,
    clip_rank_retrieval_ray,
    stage3_large_mllm_inference_qa_ray,
    stage3_large_mllm_inference_batch_cmp_ray,
)
from utils.function import get_num_of_gpu
from utils.data import load_data
import ray
import time


class xr(object):
    def __init__(self, args=None):
        super(xr, self).__init__()
        # Load dataset and image split
        self.args = args
        self.data, self.image_dict_split = load_data(args.dataset, args.dataset_path)
        
        clip_version_str = self.args.clip_path.split('/')[-1].split('-')[:4]
        self.clip_version = '-'.join(clip_version_str)
        print("Current CLIP Version: ", self.clip_version)

        self.current_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dir_name = f"runs/{args.dataset}_{args.split}/{self.clip_version}-{args.run_name}_{self.current_timestamp}"
        os.makedirs(self.dir_name, exist_ok=False)
        self.logger = create_logger(os.path.join(self.dir_name, 'output.log'))
        
        with mp.Pool(1) as pool: 
            self.num_of_gpu = pool.apply(get_num_of_gpu)

        if (self.args.dataset == 'CIRR' and self.args.subset):
            if self.args.dataset == 'CIRR':
                self.top_k = min(self.args.top_k, len(self.data['test'][0]['img_set']) - 1)
            self.logger.debug(f'Reset top k to {self.top_k}, because subset is set')
        else:
            self.top_k = self.args.top_k
        self.load_prompt()
        self.states = defaultdict(dict)

        _args_dict = vars(self.args)
        print(f"{json.dumps(_args_dict, indent=4)}")
        self.logger.info('Arguments')
        self.logger.info(f"{json.dumps(_args_dict, indent=4)}")
        self.logger.info('------------------------------------\n')
        self.logger.info(f'Current Time: {self.current_timestamp}. Start to run.')

    def load_prompt(self):
        ds_name = self.args.dataset
        with open(f'./prompts/{ds_name}/prompt1_stage1_reasoner.txt', 'r') as f:
            self.stage1_reasoner_prompt = f.read()

        # 新增：v2 结构化 prompt（可选存在）
        v2_path = f'./prompts/{ds_name}/prompt1_stage1_reasoner_v2.txt'
        self.stage1_reasoner_prompt_v2 = None
        if os.path.exists(v2_path):
            with open(v2_path, 'r') as f:
                self.stage1_reasoner_prompt_v2 = f.read()

        with open(f'./prompts/{ds_name}/prompt2_stage2_reasoner.txt', 'r') as f:
            self.stage2_reasoner_prompt = f.read()
        with open(f'./prompts/{ds_name}/caption.txt', 'r') as f:
            self.caption_prompt = f.read()
        with open(f'./prompts/{ds_name}/prompt3_stage3_evaluator.txt', 'r') as f:
            self.stage3_evaluator_prompt = f.read()

        # 新增：caption 重写 prompt（可选存在）
        rewriter_path = f'./prompts/{ds_name}/prompt1_stage1_rewriter.txt'
        self.stage1_rewriter_prompt = None
        if os.path.exists(rewriter_path):
            with open(rewriter_path, 'r') as f:
                self.stage1_rewriter_prompt = f.read()

        # 保存到本次 run 目录（便于复现）
        with open(f"{self.dir_name}/prompt1_stage1_reasoner.txt", "w") as f:
            f.write(self.stage1_reasoner_prompt)
        if self.stage1_reasoner_prompt_v2 is not None:
            with open(f"{self.dir_name}/prompt1_stage1_reasoner_v2.txt", "w") as f:
                f.write(self.stage1_reasoner_prompt_v2)
        if self.stage1_rewriter_prompt is not None:
            with open(f"{self.dir_name}/prompt1_stage1_rewriter.txt", "w") as f:
                f.write(self.stage1_rewriter_prompt)
        with open(f"{self.dir_name}/prompt2_stage2_reasoner.txt", "w") as f:
            f.write(self.stage2_reasoner_prompt)
        with open(f"{self.dir_name}/caption.txt", "w") as f:
            f.write(self.caption_prompt)
        with open(f"{self.dir_name}/prompt3_stage3_evaluator.txt", "w") as f:
            f.write(self.stage3_evaluator_prompt)



    def load_image_db(self, filter_dict):
        # Load processed image features and caption features
        image_embedding = np.load(f'./image_db/{self.args.dataset}/{self.clip_version}/image_embedding.npy')
        image_caption_embedding = np.load(f'./image_db/{self.args.dataset}/{self.clip_version}/image_caption_emebdding.npy')
        image_name = np.load(f'./image_db/{self.args.dataset}/{self.clip_version}/image_name_list.npy')

        # Filter image database
        if filter_dict is not None:
            image_name = image_name.tolist()
            filtered_image_name = []
            filtered_image_embedding = []
            filtered_image_caption_embedding = []
            for img_name, _ in tqdm(filter_dict.items(), desc='Loading Image DB'):
                idx = image_name.index(img_name)
                embedding = image_embedding[idx]
                filtered_image_name.append(img_name)
                filtered_image_embedding.append(embedding)
                filtered_image_caption_embedding.append(image_caption_embedding[idx])
        
            image_name = np.array(filtered_image_name)
            image_embedding = np.array(filtered_image_embedding)
            image_caption_embedding = np.array(filtered_image_caption_embedding)
        
        print(f'Loaded Image DB Size: {len(image_name)}')
        return image_embedding, image_caption_embedding, image_name
        
    def check_and_perpare_image_database(self):
        # Check the necessary files
        os.makedirs(f'./image_db/{self.args.dataset}/{self.clip_version}', exist_ok=True)

        # Generate the image caption if not exists
        if not os.path.exists(f'./image_db/{self.args.dataset}/image_caption.json'):
            print('Generate Image Caption...')
            mllm_query = defaultdict(dict)
            for image_name, image_path in self.image_dict_split['all'].items():
                mllm_query[image_name]['text_input'] = self.caption_prompt
                mllm_query[image_name]['image_path'] = image_path

            # Request MLLM inference
            keys = list(mllm_query.keys())
            chunk_size = math.ceil(len(keys) / self.num_of_gpu)
            key_chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
            with ray.init():
                tasks = []
                for i, key_chunk in enumerate(key_chunks):
                    query_chunk = {k: mllm_query[k] for k in key_chunk}
                    task = mllm_inference_for_caption_ray.options(num_gpus=1, num_cpus=8).remote(self.args.img_cap_model_path, query_chunk)
                    tasks.append(task)
                    time.sleep(4)
                results = ray.get(tasks)
            mllm_outputs = dict()
            for result in results:
                mllm_outputs.update(result) 
           
            image_caption_dict = {}
            for image_name, image_desc in mllm_outputs.items():
                image_caption_dict[image_name] = image_desc

            # Save image caption
            with open(f'./image_db/{self.args.dataset}/image_caption.json', 'w') as f:
                json.dump(image_caption_dict, f, indent=4)
        else:
            # If the image caption exists, load it
            with open(f'./image_db/{self.args.dataset}/image_caption.json', 'r') as f:
                image_caption_dict = json.load(f)
            
        self.image_caption_dict = image_caption_dict

        # Check if the image embedding and image caption embedding exist
        if not os.path.exists(f'./image_db/{self.args.dataset}/{self.clip_version}/image_embedding.npy') or \
            not os.path.exists(f'./image_db/{self.args.dataset}/{self.clip_version}/image_name_list.npy') or \
            not os.path.exists(f'./image_db/{self.args.dataset}/{self.clip_version}/image_caption_emebdding.npy'):
            print('Extact Image and Caption Embedding...')

            with open(f'./image_db/{self.args.dataset}/image_caption.json', 'r') as f:
                image_caption_dict = json.load(f)
            
            # Request CLIP to extact image embedding and caption embedding
            image_files = {idx: (image_name, image_path) for idx, (image_name, image_path) in enumerate(self.image_dict_split['all'].items())}
            keys = list(image_files.keys())
            chunk_size = math.ceil(len(keys) / self.num_of_gpu)
            key_chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
            with ray.init():
                tasks = []
                for i, key_chunk in enumerate(key_chunks):
                    query_chunk = {j: image_files[k] for j, k in enumerate(key_chunk)}
                    task = clip_extact_feature_ray.options(num_gpus=1, num_cpus=8).remote(self.args.clip_path, query_chunk, image_caption_dict)
                    tasks.append(task)
                results = ray.get(tasks)
            
            image_embedding = []
            image_caption_emebdding = []
            image_name_list = []
            for result in results:
                image_embedding.append(result['image_embedding'])
                image_caption_emebdding.append(result['caption_embedding'])
                image_name_list.append(result['image_name_list'])
            image_embedding = np.concatenate(image_embedding, axis=0)
            image_caption_emebdding = np.concatenate(image_caption_emebdding, axis=0)
            image_name_list = np.concatenate(image_name_list, axis=0)

            # Save image embedding
            np.save(f'./image_db/{self.args.dataset}/{self.clip_version}/image_embedding.npy', image_embedding)
            np.save(f'./image_db/{self.args.dataset}/{self.clip_version}/image_caption_emebdding.npy', image_caption_emebdding)
            np.save(f'./image_db/{self.args.dataset}/{self.clip_version}/image_name_list.npy', image_name_list)
            
        print('All files are satisfied.')

    def save_states(self, format=['pkl'], save_response_text=False, save_clip_topk=50, round_idx=None):
        print('Saving States...')
        # current_states = deepcopy(self.states)
        current_states = pickle.loads(pickle.dumps(self.states))
        for item_idx in current_states.keys():
            if 'result' in current_states[item_idx]:
                if 'stage1_retrieve' in current_states[item_idx]['result']:
                    current_states[item_idx]['result']['stage1_retrieve']['similarity'] = current_states[item_idx]['result']['stage1_retrieve']['similarity'][:save_clip_topk]
                    current_states[item_idx]['result']['stage1_retrieve']['sorted_names'] = current_states[item_idx]['result']['stage1_retrieve']['sorted_names'][:save_clip_topk]
                    current_states[item_idx]['result']['stage1_retrieve']['sorted_indices'] = current_states[item_idx]['result']['stage1_retrieve']['sorted_indices'][:save_clip_topk]
                if 's2_rerank' in current_states[item_idx]['result']:
                    current_states[item_idx]['result']['s2_rerank']['sorted_names'] = current_states[item_idx]['result']['s2_rerank']['sorted_names'][:save_clip_topk]
        
        if 'pkl' in format:
            if round_idx is None:
                pkl_filename = os.path.join(self.dir_name, 'states.pkl')
            else:
                pkl_filename = os.path.join(self.dir_name, f'states_round_{round_idx}.pkl')
            with open(pkl_filename, 'wb') as f:
                pickle.dump(current_states, f)
        if 'json' in format:
            json_filename = os.path.join(self.dir_name, 'states.json')
            with open(json_filename, 'w') as f:
                json.dump(current_states, f, indent=2)
        print('Saving States... Done')

    def calculate_recall(self, rank_result, k_list=[1,2,3,5,10,20,50,100]):
        recalls_at_K = {}
        for k in k_list:
            recalls_at_K[k] = np.mean([1 if _rnk <= k else 0 for _rnk in rank_result]).item()
        return recalls_at_K

    def stage1_reason(self, multi_round=False):
        # ===== V1：原逻辑，保持不变 =====
        llm_query = []
        for item_idx, item in enumerate(self.task_data):
            item_plan_prompt = deepcopy(self.stage1_reasoner_prompt)
            item_plan_prompt = item_plan_prompt.replace('[[INSTRUCTION]]', item['instruction'])
            if multi_round:
                ref_text = self.states[item_idx]['info']['last_round_tgt_desc'].strip()
            else:
                ref_text = self.states[item_idx]['info']['ref_img_desc'].strip()
            item_plan_prompt = item_plan_prompt.replace('[[REF_IMAGE_DESC]]', ref_text)
            llm_query.append(item_plan_prompt)

        chunk_size = math.ceil(len(llm_query) / self.num_of_gpu)
        query_chunks = [llm_query[i:i + chunk_size] for i in range(0, len(llm_query), chunk_size)]
        with ray.init():
            tasks = []
            for i, chunk in enumerate(query_chunks):
                task = llm_load_and_inference_ray.options(num_gpus=1, num_cpus=8).remote(self.args.llm_path, chunk)
                tasks.append(task)
                time.sleep(6)
            results = ray.get(tasks)
        llm_outputs = [item for (item_list, _) in results for item in item_list]

        # v1 日志
        task_time = [d for (_, d) in results]
        for _idx, d in enumerate(task_time):
            _iter_avg_time = d['time'] / d['num_task']
            self.logger.debug(f"(v1) LLM Inference {_idx}: Wall Time {d['time']:.4f}, Number Task {d['num_task']}, Avg {_iter_avg_time:.4f}")
        total_wall_time = sum([d['time'] for d in task_time])
        total_num_task = sum([d['num_task'] for d in task_time])
        iter_avg_time = total_wall_time / total_num_task if total_num_task else 0
        self.logger.debug(f"(v1) LLM Inference Overall: Wall Time {total_wall_time:.4f}, Number Task {total_num_task}, Avg {iter_avg_time:.4f}")

        # v1 解析
        for item_idx, output in enumerate(llm_outputs):
            generated_text = output.outputs[0].text
            num_input_tokens = len(output.prompt_token_ids)
            num_output_tokens = len(output.outputs[0].token_ids)

            self.states[item_idx].setdefault('result', {})
            self.states[item_idx]['result'].setdefault('stage1_reason', {})
            self.states[item_idx]['result']['stage1_reason']['num_input_tokens'] = num_input_tokens
            self.states[item_idx]['result']['stage1_reason']['num_output_tokens'] = num_output_tokens

            json_pattern = r"```json\s*([\s\S]*?)\s*```"
            match = re.search(json_pattern, generated_text, re.DOTALL)
            if match is None:
                match = re.search(r"(\{[\s\S]*?\})", generated_text, re.DOTALL)

            if match:
                json_str = match.group(1)
                json_obj = json.loads(repair_json(json_str))
                self.states[item_idx]['result']['stage1_reason']['json'] = json_obj
                if "step2" not in json_obj or len(json_obj["step2"]) == 0:
                    self.states[item_idx]['result']['stage1_reason']['json']['step2'] = [
                        self.states[item_idx]['info']['ref_img_desc']
                    ]
            else:
                # dummy v1
                self.states[item_idx]['result']['stage1_reason']['json'] = {
                    "step1": [["Modification", self.states[item_idx]['info']['inst']]],
                    "step2": [self.states[item_idx]['info']['ref_img_desc']]
                }
                self.logger.debug(f"(v1) ERROR: {generated_text}")

        # ===== V2（可选）：cir-v2 结构化输出 =====
        if getattr(self.args, 'stage1_schema', 'both') in ('v2', 'both') and self.stage1_reasoner_prompt_v2:
            llm_query_v2 = []
            for item_idx, item in enumerate(self.task_data):
                p = deepcopy(self.stage1_reasoner_prompt_v2)
                p = p.replace('[[INSTRUCTION]]', item['instruction'])
                if multi_round:
                    ref_text = self.states[item_idx]['info']['last_round_tgt_desc'].strip()
                else:
                    ref_text = self.states[item_idx]['info']['ref_img_desc'].strip()
                p = p.replace('[[REF_IMAGE_DESC]]', ref_text)
                llm_query_v2.append(p)

            chunk_size = math.ceil(len(llm_query_v2) / self.num_of_gpu)
            query_chunks = [llm_query_v2[i:i + chunk_size] for i in range(0, len(llm_query_v2), chunk_size)]
            with ray.init():
                tasks = []
                for i, chunk in enumerate(query_chunks):
                    task = llm_load_and_inference_ray.options(num_gpus=1, num_cpus=8).remote(self.args.llm_path, chunk)
                    tasks.append(task)
                    time.sleep(6)
                results_v2 = ray.get(tasks)
            llm_outputs_v2 = [item for (item_list, _) in results_v2 for item in item_list]

            for item_idx, output in enumerate(llm_outputs_v2):
                t = (output.outputs[0].text or "").strip()
                v2_obj = None
                # 先尝试裸 JSON
                try:
                    v2_obj = json.loads(repair_json(t))
                except Exception:
                    m = re.search(r"```json\s*([\s\S]*?)\s*```", t, re.DOTALL)
                    if not m:
                        m = re.search(r"(\{[\s\S]*\})", t, re.DOTALL)
                    if m:
                        try:
                            v2_obj = json.loads(repair_json(m.group(1)))
                        except:
                            pass
                if v2_obj and isinstance(v2_obj, dict):
                    self.states[item_idx]['result']['stage1_reason']['json_v2'] = v2_obj
                else:
                    self.logger.debug(f"(v2) parse fail item {item_idx}: {t[:200]}...")

        # ===== 可选：基于 v2 生成 caption（供 v2/aug 检索用） =====
        if getattr(self.args, 'stage1_rewrite_caption', False) and self.stage1_rewriter_prompt:
            self.stage1_rewrite_caption(multi_round=multi_round)

        self.save_states()
        curreret_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger.debug(f"Current Time: {curreret_time}. Target Descriptions Generated (v1 + optional v2).")


    def stage1_rewrite_caption(self, multi_round=False):
        """
        使用 cir-v2 结构化结果 + 用户指令，生成最终目标图像描述（1~2句）。
        不影响既有 v1/v2 字段，写入 result['stage1_reason']['caption_v2']。
        """
        if not self.stage1_rewriter_prompt:
            self.logger.debug("[stage1_rewrite_caption] rewriter prompt not found; skip.")
            return

        llm_query, valid_indices = [], []
        for item_idx in self.states.keys():
            s1 = self.states[item_idx]['result'].get('stage1_reason', {})
            v2 = s1.get('json_v2')
            if not v2:
                continue  # 没有 v2 就不生成 caption
            user_query = self.states[item_idx]['info']['inst']
            detail_json_str = json.dumps(v2, ensure_ascii=False, separators=(',', ': '))

            prompt = self.stage1_rewriter_prompt
            prompt = prompt.replace('{user_query}', user_query)
            prompt = prompt.replace('{detail_json_str}', detail_json_str)
            llm_query.append(prompt)
            valid_indices.append(item_idx)

        if not llm_query:
            self.logger.debug("[stage1_rewrite_caption] no items with v2; skip.")
            return

        chunk_size = math.ceil(len(llm_query) / self.num_of_gpu)
        query_chunks = [llm_query[i:i + chunk_size] for i in range(0, len(llm_query), chunk_size)]
        with ray.init():
            tasks = []
            for i, chunk in enumerate(query_chunks):
                task = llm_load_and_inference_ray.options(num_gpus=1, num_cpus=8).remote(self.args.llm_path, chunk)
                tasks.append(task)
                time.sleep(4)
            results = ray.get(tasks)
        llm_outputs = [item for (item_list, _) in results for item in item_list]

        for seq_idx, output in enumerate(llm_outputs):
            item_idx = valid_indices[seq_idx]
            text = (output.outputs[0].text or "").strip()
            # 简单清洗
            if text.startswith("```"):
                m = re.search(r"```(?:\w+)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
                if m:
                    text = m.group(1).strip()
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1].strip()
            self.states[item_idx]['result']['stage1_reason']['caption_v2'] = text

        self.save_states()
        self.logger.debug("Stage1 rewrite caption finished.")

    def stage1_retrieve(self):
        if 'CIRCO' in self.args.dataset:
            image_embedding, image_caption_embedding, image_name = self.load_image_db(None)
        else:
            image_embedding, image_caption_embedding, image_name = self.load_image_db(self.image_split)

        # ========== 解析要跑的 variant ==========
        if getattr(self.args, 'stage1_retrieve_texts', ''):
            variants = [v.strip() for v in self.args.stage1_retrieve_texts.split(',') if v.strip()]
        else:
            variants = [getattr(self.args, 'stage1_retrieve_text', 'v1')]  # 兼容老参数
        variants = [v for v in variants if v in ('v1', 'v2', 'aug')]
        if not variants:
            variants = ['v1']
        self.logger.info(f"[Stage1] Retrieval variants: {variants}  (fusion={getattr(self.args,'stage1_fusion','none')})")

        # ========== 工具函数：按 variant 设置/恢复 step2 ==========
        def _apply_variant_texts(variant):
            """临时用 variant 的文本替换 v1 的 step2，返回备份字典"""
            backup_step2 = {}
            for item_idx in self.states.keys():
                s1 = self.states[item_idx]['result'].get('stage1_reason', {})
                v1_json = s1.get('json', {})
                v1_step2 = list(v1_json.get('step2', []) or [])
                v2_cap = s1.get('caption_v2', None)

                backup_step2[item_idx] = v1_step2

                if variant == 'v1':
                    new_queries = v1_step2
                elif variant == 'v2':
                    new_queries = [v2_cap] if v2_cap else v1_step2
                else:  # 'aug' = v1 + v2
                    new_queries = v1_step2.copy()
                    if v2_cap:
                        new_queries.append(v2_cap)

                self.states[item_idx]['result']['stage1_reason']['json']['step2'] = new_queries
            return backup_step2

        def _restore_step2(backup):
            for item_idx, old in backup.items():
                try:
                    self.states[item_idx]['result']['stage1_reason']['json']['step2'] = old
                except Exception:
                    pass

        # ========== 工具函数：实际跑一次 CLIP 排序 ==========
        def _run_clip_once(tag):
            keys = list(self.states.keys())
            chunk_size = math.ceil(len(keys) / self.num_of_gpu)
            key_chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
            with ray.init():
                tasks = []
                for i, key_chunk in enumerate(key_chunks):
                    query_chunk = {k: self.states[k] for k in key_chunk}
                    task = clip_rank_retrieval_ray.options(num_gpus=1, num_cpus=8).remote(
                        self.args, query_chunk, image_embedding, image_caption_embedding, image_name
                    )
                    tasks.append(task)
                    time.sleep(5)
                results = ray.get(tasks)

            # 收集时间日志
            task_time = [d for (_, d) in results]
            for _idx, d in enumerate(task_time):
                _iter_avg_time = d['time'] / d['num_task']
                self.logger.debug(f"[Stage1-{tag}] CLIP Inference {_idx}: Wall {d['time']:.4f}, Num {d['num_task']}, Avg {_iter_avg_time:.4f}")
            total_wall_time = sum([d['time'] for d in task_time]) if task_time else 0.0
            total_num_task = sum([d['num_task'] for d in task_time]) if task_time else 0
            iter_avg_time = (total_wall_time / total_num_task) if total_num_task else 0.0
            self.logger.debug(f"[Stage1-{tag}] CLIP Inference Overall: Wall {total_wall_time:.4f}, Num {total_num_task}, Avg {iter_avg_time:.4f}")

            # 合并结果
            out = {}
            for (res, _) in results:
                out.update(res)
            return out

        # ========== 主流程：对每个 variant 跑一遍并存到专属键 ==========
        results_by_variant = {}
        for variant in variants:
            backup = _apply_variant_texts(variant)
            out = _run_clip_once(tag=variant)
            _restore_step2(backup)

            # 存入 variant 专属位置
            for item_idx in out.keys():
                self.states[item_idx]['result'][f'stage1_retrieve_{variant}'] = dict()
                for k in out[item_idx].keys():
                    self.states[item_idx]['result'][f'stage1_retrieve_{variant}'][k] = out[item_idx][k][:15000]
                # 确保存在 sorted_indices 字段，后续评估/对齐会用到
                if 'sorted_indices' not in self.states[item_idx]['result'][f'stage1_retrieve_{variant}']:
                    n = len(self.states[item_idx]['result'][f'stage1_retrieve_{variant}'].get('sorted_names', []))
                    self.states[item_idx]['result'][f'stage1_retrieve_{variant}']['sorted_indices'] = list(range(n))
            results_by_variant[variant] = out

        # ========== 可选：做一个融合榜（RRF / interleave） ==========
        fused_available = False
        fusion = getattr(self.args, 'stage1_fusion', 'none')
        if fusion != 'none' and len(variants) > 1:
            self.logger.info(f"[Stage1] Fusing rankings with method={fusion}")
            for item_idx in self.states.keys():
                # 收集多个榜单
                lists = []
                for v in variants:
                    names = self.states[item_idx]['result'][f'stage1_retrieve_{v}']['sorted_names']
                    lists.append((v, names))

                if fusion == 'rrf':
                    k_rrf = int(getattr(self.args, 'rrf_k', 60))
                    score = {}
                    for (_v, names) in lists:
                        for r, name in enumerate(names, start=1):
                            score[name] = score.get(name, 0.0) + 1.0 / (k_rrf + r)
                    fused_names = sorted(score.keys(), key=lambda n: score[n], reverse=True)
                    fused_similarity = [score[n] for n in fused_names]  # 用 RRF 分数占位
                elif fusion == 'interleave':
                    fused_names = []
                    seen = set()
                    for round_idx in range(max(len(n) for _, n in lists)):
                        for _v, names in lists:
                            if round_idx < len(names):
                                name = names[round_idx]
                                if name not in seen:
                                    fused_names.append(name)
                                    seen.add(name)
                    fused_similarity = list(range(len(fused_names), 0, -1))  # 占位分数
                else:
                    fused_names, fused_similarity = [], []

                # 写入 fused
                self.states[item_idx]['result']['stage1_retrieve_fused'] = {
                    'sorted_names': fused_names[:15000],
                    'similarity':   fused_similarity[:15000],
                    'sorted_indices': list(range(len(fused_names[:15000])))
                }
            fused_available = True

        # ========== 选择一个榜单作为“标准 stage1_retrieve”供后续阶段使用 ==========
        src = getattr(self.args, 'stage2_source', 'auto')
        if src == 'fused' and fused_available:
            chosen = 'fused'
        elif src in variants:
            chosen = src
        elif src == 'auto':
            chosen = 'fused' if fused_available else variants[0]
        else:
            chosen = variants[0]
        self.logger.info(f"[Stage1] Stage2/3 will use: {chosen}")

        # 把选中的榜单复制到兼容键 'stage1_retrieve'
        for item_idx in self.states.keys():
            key = 'stage1_retrieve_fused' if chosen == 'fused' else f'stage1_retrieve_{chosen}'
            self.states[item_idx]['result']['stage1_retrieve'] = deepcopy(self.states[item_idx]['result'][key])

        # ========== （新增）为统一键补写 tgt_img_rank，避免后续 KeyError ==========
        if self.args.split == 'valid' or (self.args.split == 'test' and self.args.dataset not in ['CIRR', 'CIRCO']):
            for item_idx in self.states.keys():
                tgt_img = self.states[item_idx]['info'].get('tgt_img')
                names = self.states[item_idx]['result']['stage1_retrieve'].get('sorted_names', [])
                if tgt_img is not None:
                    tgt_rank = names.index(tgt_img) + 1 if tgt_img in names else len(names) + 1
                    self.states[item_idx]['result']['stage1_retrieve']['tgt_img_rank'] = tgt_rank

        # ========== CIRR 子集处理（保持原逻辑） ==========
        if (self.args.dataset == 'CIRR' and self.args.subset):
            for item_idx in self.states.keys():
                subset_img_name = self.states[item_idx]['meta']['img_set']
                index_in_results = []
                for _img_name in subset_img_name:
                    if _img_name == self.states[item_idx]['info']['ref_img']:
                        continue
                    _idx = self.states[item_idx]['result']['stage1_retrieve']['sorted_names'].index(_img_name)
                    index_in_results.append(_idx)
                subset_sorted_names = [self.states[item_idx]['result']['stage1_retrieve']['sorted_names'][_idx] for _idx in index_in_results]
                subset_similarity  = [self.states[item_idx]['result']['stage1_retrieve']['similarity'][_idx] for _idx in index_in_results]
                subset_sorted_indices = [self.states[item_idx]['result']['stage1_retrieve']['sorted_indices'][_idx] for _idx in index_in_results]

                similarity_array = np.array(subset_similarity)
                sorted_indices = np.argsort(similarity_array)[::-1]
                subset_similarity  = similarity_array[sorted_indices].tolist()
                subset_sorted_names = [subset_sorted_names[i] for i in sorted_indices]
                subset_sorted_indices = [subset_sorted_indices[i] for i in sorted_indices]

                self.states[item_idx]['result']['stage1_retrieve']['sorted_names'] = subset_sorted_names
                self.states[item_idx]['result']['stage1_retrieve']['similarity']   = subset_similarity
                self.states[item_idx]['result']['stage1_retrieve']['sorted_indices'] = subset_sorted_indices

        # ========== 评估（valid/test 非 CIRR/CIRCO）— 对每个榜单和 fused 各算一份（便于对比） ==========
        def _eval_and_log(tag_key):
            if self.args.split == 'valid' or (self.args.split == 'test' and self.args.dataset not in ['CIRR','CIRCO']):
                for item_idx in self.states.keys():
                    tgt_img = self.states[item_idx]['info']['tgt_img']
                    names = self.states[item_idx]['result'][tag_key]['sorted_names']
                    tgt_img_rank = names.index(tgt_img) + 1 if tgt_img in names else len(names) + 1
                    self.states[item_idx]['result'][tag_key]['tgt_img_rank'] = tgt_img_rank
                ranks = [self.states[item_idx]['result'][tag_key]['tgt_img_rank'] for item_idx in self.states.keys()]
                metrics_recall = self.calculate_recall(ranks)
                for k in metrics_recall.keys():
                    self.logger.info(f"[{tag_key}] Recall@{k}: {metrics_recall[k] * 100:.2f}")

        # 确保存有 sorted_indices 字段并评估每个 variant
        for v in variants:
            for item_idx in self.states.keys():
                res = self.states[item_idx]['result'][f'stage1_retrieve_{v}']
                if 'sorted_indices' not in res:
                    res['sorted_indices'] = list(range(len(res.get('sorted_names', []))))
            _eval_and_log(f'stage1_retrieve_{v}')

        # 若有 fused 也评估
        if fused_available:
            _eval_and_log('stage1_retrieve_fused')

        self.save_states()
        curreret_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger.debug(f"Current Time: {curreret_time}. Clip ranking (multi-variant) done.")

    def stage2_verify(self):
        """
        生成 s2_atomic（命题集），然后按 s2_mode 计算：
        - qa:      多模态 QA，写入 result.s2_verify_qa
        - caption: caption 文本匹配，写入 result.s2_verify_caption
        - both:    两条都算，上述两个结果都会有
        为保持兼容：若 s2_mode 为单一路径，会把 result.s2_verify 指向该路径的结果。
        """
        # ===== 1) 生成 s2_atomic（与原逻辑一致） =====
        llm_query = []
        for item_idx in self.states.keys():
            atomic_inst = self.states[item_idx]['result']['stage1_reason']['json']['step1']
            atomic_inst_text = '\n'
            inst_idx = 1
            for element in atomic_inst:
                if len(element) != 2:
                    continue
                inst_type, inst_content = element
                atomic_inst_text += f"   ({inst_idx}) {inst_type}: {inst_content}\n"
                inst_idx += 1
            atomic_prompt = deepcopy(self.stage2_reasoner_prompt)
            atomic_prompt = atomic_prompt.replace('[[INSTRUCTION]]', self.states[item_idx]['info']['inst'])
            atomic_prompt = atomic_prompt.replace('[[ATOMIC_INST]]', atomic_inst_text)
            llm_query.append(atomic_prompt)

        chunk_size = max(1, math.ceil(len(llm_query) / max(self.num_of_gpu, 1)))
        query_chunks = [llm_query[i:i + chunk_size] for i in range(0, len(llm_query), chunk_size)]
        with ray.init():
            tasks = []
            for i, chunk in enumerate(query_chunks):
                task = llm_load_and_inference_ray.options(num_gpus=1, num_cpus=8).remote(self.args.llm_path, chunk)
                tasks.append(task)
                time.sleep(5)
            results = ray.get(tasks)
        llm_outputs = [item for (item_list, _) in results for item in item_list]

        task_time = [d for (_, d) in results]
        for _idx, d in enumerate(task_time):
            _iter_avg_time = d['time'] / max(d['num_task'], 1)
            self.logger.debug(f"LLM Inference {_idx}: Wall Time {d['time']:.4f}, Number Task {d['num_task']}, Avg {_iter_avg_time:.4f}")
        total_wall_time = sum([d['time'] for d in task_time]) if task_time else 0.0
        total_num_task = sum([d['num_task'] for d in task_time]) if task_time else 0
        iter_avg_time = (total_wall_time / total_num_task) if total_num_task else 0.0
        self.logger.debug(f"LLM Inference Overall: Wall Time {total_wall_time:.4f}, Number Task {total_num_task}, Avg {iter_avg_time:.4f}")

        for item_idx, output in enumerate(llm_outputs):
            generated_text = output.outputs[0].text
            self.states[item_idx]['result']['s2_atomic'] = dict()
            json_block_pat = r"```json\s*([\s\S]*?)\s*```"
            brace_pat      = r"(\{[\s\S]*?\})"
            match = re.search(json_block_pat, generated_text, re.DOTALL)
            if not match:
                match = re.search(brace_pat, generated_text, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    json_obj = json.loads(repair_json(json_str))
                except Exception as e:
                    self.logger.debug(f'{item_idx} | JSON decode fail: {e}. Raw:\n{json_str}')
                    self.logger.debug(f'{item_idx} | ERROR: {generated_text}')
                    continue
                try:
                    valid_atomic = []
                    for q_element in json_obj.get('step2', []):
                        if isinstance(q_element, list) and len(q_element) == 2:
                            valid_atomic.append(q_element)
                    json_obj['step2'] = valid_atomic
                except Exception as e:
                    self.logger.debug(f'STEP2 {item_idx} | ERROR: {e}. Text:\n{generated_text}')
                self.states[item_idx]['result']['s2_atomic']['json'] = json_obj

                # === 截断 step2 的问题数为 N ===                        
                N = getattr(self.args, "s2_num_questions", None)
                if N is not None and "step2" in json_obj:
                    step2_list = json_obj["step2"]
                    if len(step2_list) > N:
                        step2_list = step2_list[:N]
                    json_obj["step2"] = step2_list
                    self.states[item_idx]['result']['s2_atomic']['json'] = json_obj
            else:
                self.logger.debug(f'{item_idx} | ERROR: {generated_text}')

        self.save_states()
        self.logger.debug(f"Current Time: {datetime.now().strftime('%Y%m%d_%H%M%S')}. Atomic QAs ready.")

        # ===== 2) 计算不同模式的验证结果 =====
        mode = getattr(self.args, 's2_mode', 'both')

        # ---- Helper: QA 路径（MLLM 看图答题）----
        def _run_qa_path():
            mllm_query = defaultdict(dict)
            for item_idx in self.states.keys():
                if 'json' not in self.states[item_idx]['result']['s2_atomic'] or \
                'step2' not in self.states[item_idx]['result']['s2_atomic']['json']:
                    continue
                mllm_query[item_idx]['text_inputs'] = []
                mllm_query[item_idx]['ref_img_path'] = self.states[item_idx]['info']['ref_img_path']
                for q_element in self.states[item_idx]['result']['s2_atomic']['json']['step2']:
                    if not isinstance(q_element, list) or len(q_element) != 2:
                        continue
                    q_text, _ = q_element
                    mllm_query[item_idx]['text_inputs'].append(f"Answer the question with Yes or No. {q_text}")
                mllm_query[item_idx]['top_k_ranked_candidates'] = {}

                already = []
                if 's2_verify_qa' in self.states[item_idx]['result'] and \
                'candidate_satisfy_value' in self.states[item_idx]['result']['s2_verify_qa']:
                    already = list(range(len(self.states[item_idx]['result']['s2_verify_qa']['candidate_satisfy_value'])))

                for candidate_idx in range(self.top_k):
                    if candidate_idx in already:
                        continue
                    cand_name = self.states[item_idx]['result']['stage1_retrieve']['sorted_names'][candidate_idx]
                    cand_path = self.image_split[cand_name]
                    mllm_query[item_idx]['top_k_ranked_candidates'][candidate_idx] = {
                        'image_name': cand_name, 'image_path': cand_path
                    }

            if not mllm_query or not mllm_query[next(iter(mllm_query))].get('top_k_ranked_candidates', {}):
                return

            keys = list(mllm_query.keys())
            chunk_size = max(1, math.ceil(len(keys) / max(self.num_of_gpu, 1)))
            key_chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
            with ray.init():
                tasks = []
                for key_chunk in key_chunks:
                    query_chunk = {k: mllm_query[k] for k in key_chunk}
                    task = mllm_load_and_inference_ray.options(num_gpus=1, num_cpus=8).remote(self.args.mllm_path, query_chunk)
                    tasks.append(task)
                    time.sleep(6)
                results = ray.get(tasks)
            mllm_outputs = dict()
            for result, _ in results:
                mllm_outputs.update(result)

            # 聚合
            for item_idx in mllm_query.keys():
                if 's2_verify_qa' not in self.states[item_idx]['result']:
                    self.states[item_idx]['result']['s2_verify_qa'] = dict()
                cand_vals, cand_resp, cand_resp_ori = [], [], []
                for candidate_idx in range(self.top_k):
                    try:
                        already_n = len(self.states[item_idx]['result']['s2_verify_qa'].get('candidate_response_ori', [])) if 's2_verify_qa' in self.states[item_idx]['result'] else 0
                        if already_n == 0:
                            resp_ori = mllm_outputs[item_idx][candidate_idx]
                        else:
                            if candidate_idx < already_n:
                                resp_ori = self.states[item_idx]['result']['s2_verify_qa']['candidate_response_ori'][candidate_idx]
                            else:
                                resp_ori = mllm_outputs[item_idx][candidate_idx - already_n]

                        resp = ['True' if 'yes' in r.strip().lower() else 'False' for r in resp_ori]
                        gt  = [_ans for _q, _ans in self.states[item_idx]['result']['s2_atomic']['json']['step2']]
                        correct = sum(resp[i] == gt[i] for i in range(min(len(gt), len(resp))))
                        cand_vals.append(int(correct))
                        cand_resp.append(resp)
                        cand_resp_ori.append(resp_ori)
                    except Exception as e:
                        self.logger.debug(f'QA path error {item_idx}: {e}')
                        qn = len(self.states[item_idx]['result']['s2_atomic']['json'].get('step2', []))
                        cand_vals.append(0)
                        cand_resp.append(['False'] * qn)
                        cand_resp_ori.append(['No'] * qn)
                self.states[item_idx]['result']['s2_verify_qa']['candidate_satisfy_value'] = cand_vals
                self.states[item_idx]['result']['s2_verify_qa']['candidate_response'] = cand_resp
                self.states[item_idx]['result']['s2_verify_qa']['candidate_response_ori'] = cand_resp_ori

        # ---- Helper: Caption 路径（caption 与命题做文本蕴含）----
        def _run_caption_path():
            judge_template = (
                "You are a strict entailment judge.\n"
                "Given the caption:\n"
                "\"\"\"\n{CAPTION}\n\"\"\"\n"
                "Question: Does the caption support the following claim?\n"
                "\"{CLAIM}\"\n"
                "Answer STRICTLY with one word: Yes or No."
            )
            def _is_yes(text: str) -> bool:
                t = (text or "").strip().lower()
                return ("yes" in t) and ("no" not in t)

            llm_batch, llm_map = [], []  # (item_idx, cand_idx, q_idx, expected)
            for item_idx in self.states.keys():
                s2 = self.states[item_idx]['result'].get('s2_atomic', {}).get('json', {})
                qa_pairs = s2.get('step2', [])
                if not qa_pairs:
                    continue
                cand_names = self.states[item_idx]['result']['stage1_retrieve']['sorted_names'][:self.top_k]
                for cand_idx, cand_name in enumerate(cand_names):
                    cap = self.image_caption_dict.get(cand_name, "A photo.")
                    for q_idx, (q_text, q_ans) in enumerate(qa_pairs):
                        claim = q_text.strip()
                        prompt = judge_template.replace("{CAPTION}", cap).replace("{CLAIM}", claim)
                        llm_batch.append(prompt)
                        llm_map.append((item_idx, cand_idx, q_idx, bool(q_ans)))

            if llm_batch:
                chunk_size = max(1, math.ceil(len(llm_batch) / max(self.num_of_gpu, 1)))
                chunks = [llm_batch[i:i+chunk_size] for i in range(0, len(llm_batch), chunk_size)]
                with ray.init():
                    tasks = []
                    for ch in chunks:
                        tasks.append(llm_load_and_inference_ray.options(num_gpus=1, num_cpus=4).remote(self.args.llm_path, ch))
                    res = ray.get(tasks)
                llm_outputs = [item for (item_list, _) in res for item in item_list]
            else:
                llm_outputs = []

            # 初始化容器
            for item_idx in self.states.keys():
                self.states[item_idx]['result'].setdefault('s2_verify_caption', {})
                self.states[item_idx]['result']['s2_verify_caption']['candidate_satisfy_value'] = [0] * min(self.top_k, len(self.states[item_idx]['result']['stage1_retrieve']['sorted_names']))
                self.states[item_idx]['result']['s2_verify_caption']['candidate_response'] = []
                self.states[item_idx]['result']['s2_verify_caption']['candidate_response_ori'] = []

            # 聚合
            ptr = 0
            per_pred = defaultdict(lambda: defaultdict(list))  # item_idx -> cand_idx -> [bool]
            per_raw  = defaultdict(lambda: defaultdict(list))  # item_idx -> cand_idx -> [str]
            for (item_idx, cand_idx, q_idx, expected) in llm_map:
                try:
                    txt = llm_outputs[ptr].outputs[0].text
                except Exception:
                    txt = ""
                pred_yes = _is_yes(txt)
                per_pred[item_idx][cand_idx].append(pred_yes)
                per_raw[item_idx][cand_idx].append("Yes" if pred_yes else "No")
                ptr += 1

            for item_idx in self.states.keys():
                qa_pairs = self.states[item_idx]['result'].get('s2_atomic', {}).get('json', {}).get('step2', [])
                num_q = len(qa_pairs)
                cand_names = self.states[item_idx]['result']['stage1_retrieve']['sorted_names'][:self.top_k]
                cand_vals, cand_resp, cand_resp_ori = [], [], []
                for cand_idx, _ in enumerate(cand_names):
                    preds = per_pred[item_idx][cand_idx]
                    raws  = per_raw[item_idx][cand_idx]
                    correct = 0
                    for j in range(min(len(preds), num_q)):
                        expected = bool(qa_pairs[j][1])
                        ok = (preds[j] is True and expected is True) or (preds[j] is False and expected is False)
                        correct += 1 if ok else 0
                    cand_vals.append(int(correct))
                    cand_resp.append(["True" if p else "False" for p in preds[:num_q]])
                    cand_resp_ori.append(raws[:num_q])
                self.states[item_idx]['result']['s2_verify_caption']['candidate_satisfy_value'] = cand_vals
                self.states[item_idx]['result']['s2_verify_caption']['candidate_response'] = cand_resp
                self.states[item_idx]['result']['s2_verify_caption']['candidate_response_ori'] = cand_resp_ori

        # 按模式运行
        if mode in ('qa', 'both'):
            _run_qa_path()
        if mode in ('caption', 'both'):
            _run_caption_path()

        # 兼容旧字段：单一路径时把 s2_verify 指向对应结果
        if mode == 'qa':
            for item_idx in self.states.keys():
                self.states[item_idx]['result']['s2_verify'] = deepcopy(self.states[item_idx]['result'].get('s2_verify_qa', {}))
        elif mode == 'caption':
            for item_idx in self.states.keys():
                self.states[item_idx]['result']['s2_verify'] = deepcopy(self.states[item_idx]['result'].get('s2_verify_caption', {}))

        self.save_states()
        self.logger.debug(f"Current Time: {datetime.now().strftime('%Y%m%d_%H%M%S')}. Stage-2 verification finished.")


    def stage2_rerank(self):
        """
        基于三种信号分别出三份重排，并用
            【分数 = (答对题数) × (Stage-1 相似度)】
        作为排序依据：
        - s2_rerank_qa:       仅用 QA 的答对题数 × s1相似度
        - s2_rerank_caption:  仅用 Caption 的答对题数 × s1相似度
        - s2_rerank_fusion:   (0.5*QA答对数 + 0.5*Caption答对数) × s1相似度
        兼容：若 s2_mode 单一路径，则把 result.s2_rerank 指向对应的那份；
            若 s2_mode='both'，把 result.s2_rerank 指向 fusion。
        注意：Stage-1 相似度来自 result.stage1_retrieve.similarity（与 sorted_names 对齐）。
        若相似度可能为负，你可以在下面加一行进行截断或归一化。
        """
        # def _take_head(item_idx):
        #     names_all = self.states[item_idx]['result']['stage1_retrieve']['sorted_names']
        #     sims_all  = self.states[item_idx]['result']['stage1_retrieve'].get('similarity', [])
        #     head_names = names_all[:self.top_k]
        #     # 与 head 对齐的 s1 相似度
        #     head_sims = (sims_all[:len(head_names)] if sims_all else [1.0]*len(head_names))
        #     # 如果你担心负值影响，可启用截断：
        #     head_sims = [max(0.0, s) for s in head_sims]
        #     return names_all, head_names, head_sims

        def _take_head(item_idx):
            names_all = self.states[item_idx]['result']['stage1_retrieve']['sorted_names']
            sims_all  = self.states[item_idx]['result']['stage1_retrieve'].get('similarity', [])
            head_names = names_all[:self.top_k]
            head_sims = (sims_all[:len(head_names)] if sims_all else [1.0] * len(head_names))

            # ---- 最小-最大归一化到 [0,1] ----
            if head_sims:  # 避免空列表
                lo, hi = min(head_sims), max(head_sims)
                if hi > lo:  # 正常情况
                    head_sims = [(x - lo) / (hi - lo) for x in head_sims]
                else:
                    # 如果全一样，直接设为全 0 或全 1，这里给全 0
                    head_sims = [0.0] * len(head_sims)

            return names_all, head_names, head_sims


        # def _take_head(item_idx):
        #     names_all = self.states[item_idx]['result']['stage1_retrieve']['sorted_names']
        #     sims_all  = self.states[item_idx]['result']['stage1_retrieve'].get('similarity', [])
        #     head_names = names_all[:self.top_k]
        #     head_sims = (sims_all[:len(head_names)] if sims_all else [1.0] * len(head_names))

        #     # ---- softmax 归一化 ----
        #     if head_sims:
        #         m = max(head_sims)  # 数值稳定性：先减去最大值
        #         exps = [math.exp(x - m) for x in head_sims]
        #         s = sum(exps)
        #         head_sims = [e / s for e in exps]

        #     return names_all, head_names, head_sims

        def _sort_by_combined(head_names, head_scores, tail_names):
            order = np.argsort(-np.array(head_scores))
            re_head = [head_names[i] for i in order.tolist()]
            return re_head + tail_names

        def _do_single_rerank(item_idx, verify_key, out_key):
            names_all, head_names, head_sims = _take_head(item_idx)
            if verify_key not in self.states[item_idx]['result']:
                return
            vals = self.states[item_idx]['result'][verify_key].get('candidate_satisfy_value', [])
            vals = list(vals[:len(head_names)]) if vals is not None else [0]*len(head_names)
            head_scores = (np.array(vals, dtype=float) * np.array(head_sims, dtype=float)).tolist()
            new_sorted = _sort_by_combined(head_names, head_scores, names_all[self.top_k:])
            self.states[item_idx]['result'].setdefault(out_key, {})['sorted_names'] = new_sorted

        def _do_fusion_rerank(item_idx, out_key):
            names_all, head_names, head_sims = _take_head(item_idx)
            num_q = len(self.states[item_idx]['result'].get('s2_atomic', {}).get('json', {}).get('step2', [])) or 1

            qa_vals  = self.states[item_idx]['result'].get('s2_verify_qa', {}).get('candidate_satisfy_value', [])
            cap_vals = self.states[item_idx]['result'].get('s2_verify_caption', {}).get('candidate_satisfy_value', [])
            qa_vals  = np.array(qa_vals[:len(head_names)] + [0]*max(0, len(head_names)-len(qa_vals)), dtype=float)
            cap_vals = np.array(cap_vals[:len(head_names)] + [0]*max(0, len(head_names)-len(cap_vals)), dtype=float)

            # 融合的“数量”：等权平均（若想改成求和/加权/accuracy，可自行替换）
            fused_count = 0.5*qa_vals + 0.5*cap_vals
            head_scores = (fused_count * np.array(head_sims, dtype=float)).tolist()
            new_sorted = _sort_by_combined(head_names, head_scores, names_all[self.top_k:])
            self.states[item_idx]['result'].setdefault(out_key, {})['sorted_names'] = new_sorted

        # 逐样本生成三份榜单（视可用性）
        for item_idx in tqdm(self.states.keys(), desc='Reranking (qa/caption/fusion × s1-sim)', total=len(self.states)):
            # QA-only
            if 's2_verify_qa' in self.states[item_idx]['result']:
                _do_single_rerank(item_idx, 's2_verify_qa', 's2_rerank_qa')
            # Caption-only
            if 's2_verify_caption' in self.states[item_idx]['result']:
                _do_single_rerank(item_idx, 's2_verify_caption', 's2_rerank_caption')
            # Fusion（需要至少一条存在；不存在的置 0）
            if ('s2_verify_qa' in self.states[item_idx]['result']) or ('s2_verify_caption' in self.states[item_idx]['result']):
                _do_fusion_rerank(item_idx, 's2_rerank_fusion')

        # 兼容旧字段：根据 s2_mode 选择一个别名 s2_rerank
        mode = getattr(self.args, 's2_mode', 'both')
        for item_idx in self.states.keys():
            if mode == 'qa' and 's2_rerank_qa' in self.states[item_idx]['result']:
                self.states[item_idx]['result']['s2_rerank'] = deepcopy(self.states[item_idx]['result']['s2_rerank_qa'])
            elif mode == 'caption' and 's2_rerank_caption' in self.states[item_idx]['result']:
                self.states[item_idx]['result']['s2_rerank'] = deepcopy(self.states[item_idx]['result']['s2_rerank_caption'])
            elif mode == 'both' and 's2_rerank_fusion' in self.states[item_idx]['result']:
                self.states[item_idx]['result']['s2_rerank'] = deepcopy(self.states[item_idx]['result']['s2_rerank_fusion'])

        # 评估（若可评估）
        def _eval_and_log(tag):
            if self.args.split == 'valid' or (self.args.split == 'test' and self.args.dataset not in ['CIRR','CIRCO']):
                for item_idx in self.states.keys():
                    tgt_img = self.states[item_idx]['info']['tgt_img']
                    ranked = self.states[item_idx]['result'].get(tag, {}).get('sorted_names', [])
                    tgt_rank = ranked.index(tgt_img) + 1 if tgt_img in ranked else len(ranked) + 1
                    self.states[item_idx]['result'].setdefault(tag, {})['tgt_img_rank'] = tgt_rank
                ranks = [self.states[item_idx]['result'].get(tag, {}).get('tgt_img_rank', 1e9) for item_idx in self.states.keys()]
                metrics_recall = self.calculate_recall(ranks)
                for k in metrics_recall.keys():
                    self.logger.info(f"[{tag}] Recall@{k}: {metrics_recall[k] * 100:.2f}")

        for tag in ['s2_rerank_qa', 's2_rerank_caption', 's2_rerank_fusion', 's2_rerank']:
            any_exist = any(tag in self.states[i]['result'] for i in self.states.keys())
            if any_exist:
                _eval_and_log(tag)

        self.save_states()
        self.logger.debug(f"Current Time: {datetime.now().strftime('%Y%m%d_%H%M%S')}. Stage-2 rerank (× s1-sim) finished.")


    def stage3_evaluate(self):
        """
        Stage 3: 使用大模型对 top-α 候选进行成对比较，将判定为“满足指令(Yes)”的候选
        直接前置到列表最前面，其余保持原相对顺序不变。
        该版本修复了：
        1) 判决解析过于严格导致从不触发的问题（新增稳健的 Yes/No 解析）；
        2) 使用 pop 引发的索引错位问题（改为一次性重组列表）；
        3) 仅在必要时重新请求评估模型；并增加若干日志以便排查。
        """
        import re

        def _is_yes(text: str) -> bool:
            """稳健判断是否为 Yes/True（避免误判包含 no/false 的句子）"""
            t = (text or "").strip().lower()
            yes = re.search(r'\b(yes|true|是|对|正确)\b', t) is not None
            no  = re.search(r'\b(no|false|否|不|错)\b', t) is not None
            return yes and not no

        # 1) 构造评估查询（仅当没有已有的 stage3_eval 输出时才会跑一遍模型）
        need_query = True
        try:
            if 'stage3_eval' in self.states[0]['result'] and \
            'outputs' in self.states[0]['result']['stage3_eval']:
                need_query = False
        except Exception:
            need_query = True

        if need_query:
            mllm_query = defaultdict(dict)
            for item_idx in self.states.keys():
                mllm_query[item_idx]['top_candidate'] = []
                mllm_query[item_idx]['inst'] = self.states[item_idx]['info']['inst']
                mllm_query[item_idx]['ref_img_path'] = self.states[item_idx]['info']['ref_img_path']

                # 更强的输出约束：只输出 Yes 或 No
                text_input = deepcopy(self.stage3_evaluator_prompt)
                text_input = text_input.replace(
                    '[[INSTRUCTION]]',
                    mllm_query[item_idx]['inst'].strip(' .').lower()
                )
                # 强约束：只输出一个词
                text_input += (
                    "\n\nIMPORTANT: Respond with exactly one word: Yes or No. "
                    "Do not output any other words, symbols, or explanations."
                )

                # 只取前 alpha 个候选做成对比较
                alpha = min(self.args.alpha, len(self.states[item_idx]['result']['s2_rerank']['sorted_names']))
                for idx in range(alpha):
                    candidate_img_name = self.states[item_idx]['result']['s2_rerank']['sorted_names'][idx]
                    candidate_img_path = self.image_split[candidate_img_name]
                    mllm_query[item_idx]['top_candidate'].append({
                        'image_name': candidate_img_name,
                        'image_path': candidate_img_path
                    })

                mllm_query[item_idx]['text_input'] = text_input

            # 2) 切分任务，调用评估模型
            keys = list(mllm_query.keys())
            chunk_size = math.ceil(len(keys) / self.num_of_gpu)
            key_chunks = [keys[i:i + chunk_size] for i in range(0, len(keys), chunk_size)]
            with ray.init():
                tasks = []
                for i, key_chunk in enumerate(key_chunks):
                    query_chunk = {k: mllm_query[k] for k in key_chunk}
                    task = stage3_large_mllm_inference_batch_cmp_ray.options(
                        num_gpus=1,
                        num_cpus=8
                    ).remote(self.args.eval_mllm_path, query_chunk)
                    tasks.append(task)
                    time.sleep(5)
                results = ray.get(tasks)

            mllm_outputs = dict()
            for result, _ in results:
                mllm_outputs.update(result)

            # 记录时间日志
            task_time = [d for (_, d) in results]
            for _idx, d in enumerate(task_time):
                _iter_avg_time = d['time'] / max(d['num_task'], 1)
                self.logger.debug(f"MLLM Inference {_idx}: Wall Time {d['time']:.4f}, "
                                f"Number Task {d['num_task']}, Avg {_iter_avg_time:.4f}")
            if task_time and 'turn_time' in task_time[0]:
                num_turn = len(task_time[0]['turn_time'])
                for turn_idx in range(num_turn):
                    for _idx, d in enumerate(task_time):
                        _turn_avg_time = d['turn_time'][turn_idx] / max(d['turn_task'][turn_idx], 1)
                        self.logger.debug(f"MLLM Inference {_idx} Turn {turn_idx+1}: "
                                        f"Wall Time {d['turn_time'][turn_idx]:.4f}, "
                                        f"Number Task {d['turn_task'][turn_idx]}, "
                                        f"Avg {_turn_avg_time:.4f}")
            total_wall_time = sum([d['time'] for d in task_time]) if task_time else 0.0
            total_num_task = sum([d['num_task'] for d in task_time]) if task_time else 0
            iter_avg_time = (total_wall_time / total_num_task) if total_num_task else 0.0
            self.logger.debug(f"MLLM Inference Overall: Wall Time {total_wall_time:.4f}, "
                            f"Number Task {total_num_task}, Avg {iter_avg_time:.4f}")

        # 3) 根据评估结果重排
        moved_cnt = 0
        decided_yes_cnt = 0
        total_items = len(self.states)

        for item_idx in tqdm(self.states.keys(), desc='[Stage3] Evaluating'):
            # 如果没有 outputs，则填充
            if 'stage3_eval' not in self.states[item_idx]['result']:
                self.states[item_idx]['result']['stage3_eval'] = dict()

            if 'outputs' not in self.states[item_idx]['result']['stage3_eval']:
                # 只有在上面跑了模型的情况下才有 mllm_outputs
                if need_query:
                    self.states[item_idx]['result']['stage3_eval']['outputs'] = mllm_outputs.get(item_idx, [])
                else:
                    # 既然无需重新跑，那就保持已有 outputs
                    self.states[item_idx]['result']['stage3_eval']['outputs'] = \
                        self.states[item_idx]['result']['stage3_eval'].get('outputs', [])

            outputs = self.states[item_idx]['result']['stage3_eval']['outputs']

            # 原始（S2 后）的排序
            sorted_names = list(self.states[item_idx]['result']['s2_rerank']['sorted_names'])
            alpha = min(self.args.alpha, len(outputs), len(sorted_names))

            # 收集前 α 个 turn 中被判定为 "Yes" 的索引
            hit_idx = []
            for turn_idx in range(alpha):
                tr = outputs[turn_idx] if turn_idx < len(outputs) else {}
                decision = tr.get('decision', None)
                if decision is None:
                    # 兜底：根据文本再解析一次
                    decision = _is_yes(tr.get('text', ''))
                if decision:
                    hit_idx.append(turn_idx)

            decided_yes_cnt += (1 if len(hit_idx) > 0 else 0)

            # 重排：把命中的放到最前面，其余保持原相对顺序
            if hit_idx:
                hit_set = set(hit_idx)
                front = [sorted_names[i] for i in hit_idx]
                remain = [img for i, img in enumerate(sorted_names) if i not in hit_set]
                evaluated_sorted_names = front + remain
                moved_cnt += 1
            else:
                evaluated_sorted_names = sorted_names

            self.states[item_idx]['result']['stage3_eval']['sorted_names'] = evaluated_sorted_names

        self.logger.info(f"[Stage3] items with at least one YES in top-{self.args.alpha}: "
                        f"{decided_yes_cnt}/{total_items}")
        self.logger.info(f"[Stage3] items reordered (moved front): {moved_cnt}/{total_items}")

        # 4) 统计指标（与原逻辑一致）
        if self.args.split == 'valid' or (self.args.split == 'test' and self.args.dataset not in ['CIRR', 'CIRCO']):
            for item_idx in self.states.keys():
                tgt_img = self.states[item_idx]['info']['tgt_img']
                ranked = self.states[item_idx]['result']['stage3_eval']['sorted_names']
                if tgt_img in ranked:
                    tgt_img_rank = ranked.index(tgt_img) + 1
                else:
                    tgt_img_rank = len(ranked) + 1
                self.states[item_idx]['result']['stage3_eval']['tgt_img_rank'] = tgt_img_rank

            rank_results = [self.states[item_idx]['result']['stage3_eval']['tgt_img_rank']
                            for item_idx in self.states.keys()]
            metrics_recall = self.calculate_recall(rank_results)
            for k in metrics_recall.keys():
                self.logger.info(f"Recall@{k}: {metrics_recall[k] * 100}")

        self.save_states()
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger.debug(f"Current Time: {current_time}. [Stage3] Evaluation is finished.")


    def start(self):
        if self.args.split == 'valid':
            self.task_data = self.data['val']
            self.image_split = self.image_dict_split['val']
        else:
            self.task_data = self.data['test']
            self.image_split = self.image_dict_split['test']
        
        if self.args.restore_states is not None:
            if os.path.exists(self.args.restore_states):
                self.logger.info(f"Restoring status from {self.args.restore_states}")
                self.states = pickle.load(open(self.args.restore_states, 'rb'))
            else:
                raise FileNotFoundError(f"Status file {self.args.restore_status} does not exist")
        
        self.check_and_perpare_image_database()

        for item_idx, item in enumerate(self.task_data):
            if 'CIRR' in self.args.dataset:
                self.states[item_idx]['meta'] = dict()
                self.states[item_idx]['meta']['pairid'] = item['pairid'] 
                self.states[item_idx]['meta']['img_set'] = item['img_set']
            elif 'CIRCO' in self.args.dataset:
                self.states[item_idx]['meta'] = dict()
                self.states[item_idx]['meta']['pairid'] = item['id'] 

            # save to states
            self.states[item_idx]['info'] = dict()
            self.states[item_idx]['info']['ref_img'] = item['ref']
            self.states[item_idx]['info']['ref_img_path'] = self.image_split[item['ref']] if item['ref'] is not None else None
            self.states[item_idx]['info']['ref_img_desc'] = self.image_caption_dict[item['ref']] if item['ref'] is not None else "A blank image."
            self.states[item_idx]['info']['inst'] = item['instruction']

            if self.args.split == 'test' and self.args.dataset in ['CIRR', 'CIRCO']:
                pass
            else:
                self.states[item_idx]['info']['tgt_img'] = item['tgt']
                self.states[item_idx]['info']['tgt_img_path'] = self.image_split[item['tgt']]
       
        ###############################
        ###                         ###
        ###############################
        if 's1' in self.args.stages:
            if self.args.restore_states is None or \
                (self.args.restore_states is not None and self.args.stage_skip_num <= 0):
                self.stage1_reason()
            if self.args.restore_states is None or \
                (self.args.restore_states is not None and self.args.stage_skip_num <= 1):
                self.stage1_retrieve()
        if 's2' in self.args.stages:
            if self.args.restore_states is None or \
                (self.args.restore_states is not None and self.args.stage_skip_num <= 2):
                self.stage2_verify()
            if self.args.restore_states is None or \
                (self.args.restore_states is not None and self.args.stage_skip_num <= 3):
                self.stage2_rerank()
        if 's3' in self.args.stages:
            if self.args.restore_states is None or \
                (self.args.restore_states is not None and self.args.stage_skip_num <= 4):
                self.stage3_evaluate()

        ###############################
        ###                         ###
        ###############################

        if self.args.split == 'test' and self.args.dataset in ['CIRR', 'CIRCO']:
            self.logger.info(f"Genereating test results...")
            test_results = dict()
            defatut_submit_top_k = 50
            result_file_name = f"{self.current_timestamp}_{self.args.dataset}_{self.args.split}_stage1_retrieve.json"
            if 'CIRR' in self.args.dataset:
                if self.args.subset:
                    test_results['version'] = 'rc2'
                    test_results['metric'] = 'recall_subset'
                    defatut_submit_top_k = 3
                    result_file_name = f"{self.current_timestamp}_{self.args.dataset}_subset_{self.args.split}_clip.json"
                else:
                    test_results['version'] = 'rc2'
                    test_results['metric'] = 'recall'
            

            for item_idx in self.states.keys():
                pairid = self.states[item_idx]['meta']['pairid']
                top_50_prediction = self.states[item_idx]['result']['stage1_retrieve']['sorted_names'][:defatut_submit_top_k]
                test_results[pairid] = top_50_prediction
                if self.args.dataset == 'CIRCO':
                    for idx, pred in enumerate(test_results[pairid]):
                            # remove .jpg and zeros in th beginging
                            pred = pred[:-4].lstrip('0')
                            test_results[pairid][idx] = pred
            
            self.logger.info(f"Saving test results to {os.path.join(self.dir_name, result_file_name)}")
            with open(os.path.join(self.dir_name, result_file_name), 'w') as f:
                json.dump(test_results, f)

            if "s2_rerank" in self.states[item_idx]['result'].keys():
                test_results = dict()
                if self.args.dataset == 'CIRR':
                    test_results['version'] = 'rc2'
                    test_results['metric'] = 'recall' if not self.args.subset else 'recall_subset'
                for item_idx in self.states.keys():
                    pairid = self.states[item_idx]['meta']['pairid']
                    top_50_prediction = self.states[item_idx]['result']['s2_rerank']['sorted_names'][:defatut_submit_top_k]
                    test_results[pairid] = top_50_prediction
                    if self.args.dataset == 'CIRCO':
                        for idx, pred in enumerate(test_results[pairid]):
                            # remove .jpg and zeros in th beginging
                            pred = pred[:-4].lstrip('0')
                            test_results[pairid][idx] = pred
                rerank_file_name = f"{self.current_timestamp}_{self.args.dataset}_{self.args.split}_stage2_rerank.json"
                self.logger.info(f"Saving rerank results to {os.path.join(self.dir_name, rerank_file_name)}")
                with open(os.path.join(self.dir_name, rerank_file_name), 'w') as f:
                    json.dump(test_results, f)

            if "stage3_eval" in self.states[item_idx]['result'].keys():
                test_results = dict()
                if self.args.dataset == 'CIRR':
                    test_results['version'] = 'rc2'
                    test_results['metric'] = 'recall' if not self.args.subset else 'recall_subset'
                for item_idx in self.states.keys():
                    pairid = self.states[item_idx]['meta']['pairid']
                    top_50_prediction = self.states[item_idx]['result']['stage3_eval']['sorted_names'][:defatut_submit_top_k]
                    test_results[pairid] = top_50_prediction
                    if self.args.dataset == 'CIRCO':
                        for idx, pred in enumerate(test_results[pairid]):
                            # remove .jpg and zeros in th beginging
                            pred = pred[:-4].lstrip('0')
                            test_results[pairid][idx] = pred
                rerank_file_name = f"{self.current_timestamp}_{self.args.dataset}_{self.args.split}_stage3_eval.json"
                self.logger.info(f"Saving rerank results to {os.path.join(self.dir_name, rerank_file_name)}")
                with open(os.path.join(self.dir_name, rerank_file_name), 'w') as f:
                    json.dump(test_results, f)
            
        elif self.args.split == 'valid':
            max_logged_case = 10
            cnt = 0
            for item_idx in self.states.keys():
                # previous_rank = self.states[item_idx]['result']['stage1_retrieve']['tgt_img_rank']
                # verified_rank = self.states[item_idx]['result']['s2_rerank']['tgt_img_rank']
                # rank_change = previous_rank - verified_rank
                if True:
                # if previous_rank <= self.topk and rank_change <= 0 and verified_rank != 1:
                    cnt += 1
                    self.logger.debug(f'-------{item_idx}-------')
                    self.logger.debug("# Query")
                    self.logger.debug("## Instruction\n - {}".format(self.states[item_idx]['info']['inst']))
                    self.logger.debug("## Reference Image Description\n - {}".format(self.states[item_idx]['info']['ref_img_desc']))
                    self.logger.debug("Ref Img Path: {}".format(self.states[item_idx]['info']['ref_img_path']))
                    if self.args.split == 'valid':
                        self.logger.debug("Tgt Img Path: {}".format(self.states[item_idx]['info']['tgt_img_path']))
                        self.logger.debug("Tgt Img Desc: {}".format(self.image_caption_dict[self.states[item_idx]['info']['tgt_img']]))
                    self.logger.debug('## Target Image Description')
                    if 'json' in self.states[item_idx]['result']['stage1_reason'].keys():
                        # self.logger.debug(json.dumps(self.states[item_idx]['result']['stage1_reason']['json'], indent=2))
                        for d_idx, d in enumerate(self.states[item_idx]['result']['stage1_reason']['json']['step1']):
                            self.logger.debug(f" ({d_idx+1}) {d}")
                    self.logger.debug('## Verification questions')
                    if 's2_atomic' in self.states[item_idx]['result'].keys() and 'json' in self.states[item_idx]['result']['s2_atomic'].keys():
                        # self.logger.debug(json.dumps(self.states[item_idx]['result']['s2_atomic']['json'], indent=2))
                        for q_idx, (q, a) in enumerate(self.states[item_idx]['result']['s2_atomic']['json']['step2']):
                            self.logger.debug(f" ({q_idx+1}) {q} Ground Truth: {a}.")
                    self.logger.debug('# Candidates')
                    for candidate_idx in range(self.top_k):
                        img_name = self.states[item_idx]['result']['stage1_retrieve']['sorted_names'][candidate_idx]
                        img_path = self.image_split[img_name]
                        img_desc = self.image_caption_dict[img_name]
                        similarity_score = self.states[item_idx]['result']['stage1_retrieve']['similarity'][candidate_idx]
                        is_ground_truth = candidate_idx + 1 == self.states[item_idx]['result']['stage1_retrieve']['tgt_img_rank']
                        ground_truth_flag = '  <- Ground truth' if is_ground_truth else ''
                        self.logger.debug(f"## Candidate {candidate_idx+1} {ground_truth_flag}")
                        self.logger.debug(" - Similarity: {:.4f}".format(similarity_score))
                        self.logger.debug(" - Image description: {}".format(img_desc))
                        if 's2_verify' in self.states[item_idx]['result'].keys():
                            self.logger.debug(" - Response of Questions: {}".format(self.states[item_idx]['result']['s2_verify']['candidate_response'][candidate_idx]))
                            _gt = [x for _, x in self.states[item_idx]['result']['s2_atomic']['json']['step2']]
                            self.logger.debug("  Ground truth: {}".format(_gt))
                            self.logger.debug('  Correct: {}'.format(self.states[item_idx]['result']['s2_verify']['candidate_satisfy_value'][candidate_idx]))
                            self.logger.debug(" - Candidate Image {}".format(img_path))
                    self.logger.debug("Previous reank {}".format(self.states[item_idx]['result']['stage1_retrieve']['tgt_img_rank']))
                    # self.logger.debug("New rerank {}".format(self.states[item_idx]['result']['s2_rerank']['tgt_img_rank']))
                    # self.logger.debug("Rank change: {}".format(rank_change))
                    self.logger.debug(f'-------{item_idx}-------')
                if cnt >= max_logged_case:
                    break
    
    def start_dialog(self):
        if self.args.split == 'valid':
            self.task_data = self.data['val']
            self.image_split = self.image_dict_split['val']
        else:
            self.task_data = self.data['test']
            self.image_split = self.image_dict_split['test']
        
        if self.args.restore_states is not None:
            if os.path.exists(self.args.restore_states):
                self.logger.info(f"Restoring status from {self.args.restore_states}")
                self.states = pickle.load(open(self.args.restore_states, 'rb'))
            else:
                raise FileNotFoundError(f"Status file {self.args.restore_status} does not exist")
        
        self.check_and_perpare_image_database()

        ###############################
        ###                         ###
        ###############################
        task_data = deepcopy(self.task_data)
        total_round = len(task_data[0]['dialog_list'])
        current_round_task = dict()
        k_list = [1,2,3,4,5,10,20,25,50]
        stage_hits_rate = {_: dict() for _ in [1,2,3]}
        for stage_idx in stage_hits_rate.keys():
            stage_hits_rate[stage_idx] = {
                k: [0] * len(task_data) for k in k_list
            }
        
        for round_idx in range(1, total_round):
            self.logger.info(f"Starting round {round_idx}")
            
            states = {}
            current_round_task_data = []
            for item_idx in range(len(task_data)):
                if round_idx == 1:
                    last_round_tgt_desc = task_data[item_idx]['dialog_list'][0]
                else:
                    if 'json' in self.states[item_idx]['result']['stage1_reason'].keys() and len(self.states[item_idx]['result']['stage1_reason']['json']['step2']):
                        last_round_tgt_desc = self.states[item_idx]['result']['stage1_reason']['json']['step2'][0]
                
                current_round_task_data.append({
                    'ref': None,
                    'tgt': task_data[item_idx]['tgt'],
                    'instruction': task_data[item_idx]['dialog_list'][round_idx],
                    'last_round_tgt_desc': last_round_tgt_desc,
                })

                if item_idx <= 5:
                    s = f"{item_idx}: " +  "last_round_tgt_desc: " + last_round_tgt_desc +  ". instruction: " + task_data[item_idx]['dialog_list'][round_idx]
                    self.logger.info(s)

            self.states = defaultdict(dict)
            self.task_data = current_round_task_data

            for item_idx, item in enumerate(self.task_data):
                self.states[item_idx]['info'] = dict()
                self.states[item_idx]['info']['ref_img'] = item['ref']
                self.states[item_idx]['info']['ref_img_path'] = self.image_split[item['ref']] if item['ref'] is not None else None
                self.states[item_idx]['info']['ref_img_desc'] = self.image_caption_dict[item['ref']] if item['ref'] is not None else "A blank image."
                self.states[item_idx]['info']['inst'] = item['instruction']
                self.states[item_idx]['info']['last_round_tgt_desc'] = item['last_round_tgt_desc']
                self.states[item_idx]['info']['tgt_img'] = item['tgt']
                self.states[item_idx]['info']['tgt_img_path'] = self.image_split[item['tgt']]
        
            ###############################
            ###                         ###
            ###############################
            if 's1' in self.args.stages:
                if self.args.restore_states is None or \
                    (self.args.restore_states is not None and self.args.stage_skip_num <= 0):
                    self.stage1_reason(multi_round=True)
                if self.args.restore_states is None or \
                    (self.args.restore_states is not None and self.args.stage_skip_num <= 1):
                    self.stage1_retrieve()
                
                self.logger.info(f"Stage 1 Hits rate.")
                tgt_img_ranks = [self.states[item_idx]['result']['stage1_retrieve']['tgt_img_rank'] for item_idx in self.states.keys()]
                for k in k_list:
                    for rnk_idx, rnk in enumerate(tgt_img_ranks):
                        if stage_hits_rate[1][k][rnk_idx]:
                            continue
                        else:
                            if rnk <= k:
                                stage_hits_rate[1][k][rnk_idx] += 1
                    self.logger.info(f"Hits@{k}: {sum(stage_hits_rate[1][k]) / len(stage_hits_rate[1][k])}")


            if 's2' in self.args.stages:
                if self.args.restore_states is None or \
                    (self.args.restore_states is not None and self.args.stage_skip_num <= 2):
                    self.stage2_verify()
                if self.args.restore_states is None or \
                    (self.args.restore_states is not None and self.args.stage_skip_num <= 3):
                    self.stage2_rerank()

                self.logger.info(f"Stage 2 Hits rate.")
                tgt_img_ranks = [self.states[item_idx]['result']['s2_rerank']['tgt_img_rank'] for item_idx in self.states.keys()]
                for k in k_list:
                    for rnk_idx, rnk in enumerate(tgt_img_ranks):
                        if stage_hits_rate[2][k][rnk_idx]:
                            continue
                        else:
                            if rnk <= k:
                                stage_hits_rate[2][k][rnk_idx] += 1
                    self.logger.info(f"Hits@{k}: {sum(stage_hits_rate[2][k]) / len(stage_hits_rate[2][k])}")

            if 's3' in self.args.stages:
                if self.args.restore_states is None or \
                    (self.args.restore_states is not None and self.args.stage_skip_num <= 4):
                    self.stage3_evaluate()

                self.logger.info(f"Stage 3 Hits rate.")
                tgt_img_ranks = [self.states[item_idx]['result']['stage3_eval']['tgt_img_rank'] for item_idx in self.states.keys()]
                for k in k_list:
                    for rnk_idx, rnk in enumerate(tgt_img_ranks):
                        if stage_hits_rate[3][k][rnk_idx]:
                            continue
                        else:
                            if rnk <= k:
                                stage_hits_rate[3][k][rnk_idx] += 1
                    self.logger.info(f"Hits@{k}: {sum(stage_hits_rate[3][k]) / len(stage_hits_rate[3][k])}")
            
            self.save_states(round_idx=round_idx)
            
            ###############################
            ###                         ###
            ###############################


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--llm_path', type=str, default="OpenGVLab/InternVL3-8B", help='Reasoner model path')
    parser.add_argument('--mllm_path', type=str, default="OpenGVLab/InternVL3-8B", help='Verifier model path')
    parser.add_argument('--img_cap_model_path', type=str, default="OpenGVLab/InternVL3-8B", help='Captioner model path')
    parser.add_argument('--eval_mllm_path', type=str, default="OpenGVLab/InternVL3-8B", help='Evaluator model path')
    parser.add_argument('--clip_path', type=str, default="laion/CLIP-ViT-B-32-laion2B-s34B-b79K", help='VLM path')
    parser.add_argument('--dataset_path', type=str, default="data/", help='Dataset folder path')

    parser.add_argument('--stage1_retrieve_texts', type=str, default='v1,v2,aug',
                        help='Comma-separated multiple sources to retrieve in parallel, e.g., "v1,v2,aug".')
    parser.add_argument('--stage1_fusion', type=str, choices=['none','rrf','interleave'], default='rrf',
                        help='Fuse multiple retrieval lists into a single fused list.')
    parser.add_argument('--rrf_k', type=int, default=60, help='RRF constant k for reciprocal-rank fusion.')

    parser.add_argument('--stage2_source', type=str, default='aug',
                        choices=['auto','v1','v2','aug','fused'],
                        help='Which stage1 list to feed into stage2/3. "auto"=use fused if available else the first variant.')

    parser.add_argument('--run_name', type=str, default='default')
    
    parser.add_argument('--stage1_schema', type=str, choices=['v1', 'v2', 'both'], default='both',
                        help='Stage-1 outputs: v1 (step1/step2), v2 (cir-v2), or both.')
    
    parser.add_argument('--stage1_rewrite_caption', action='store_true', default=True,
                        help='After stage1, rewrite a final target caption using the cir-v2 detail set.')

    parser.add_argument("--s2_num_questions", type=int, default=None, help="限制 Stage2 生成的问题数量（默认不限制）")
    parser.add_argument('--s2_mode', type=str, choices=['qa','caption','both'], default='both', help='Stage-2 mode: qa=MLLM-QA, caption=caption-only, both=run both and also fusion')

    parser.add_argument('--tau', type=float, default=0.15)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--alpha', type=int, default=100)
    parser.add_argument('--stages', nargs='+', choices=['s1', 's2', 's3'], default=['s1', 's2'])

    parser.add_argument('--dataset', type=str, choices=[
        'CIRR', 'CIRCO', 
        'FashionIQ-dress', 'FashionIQ-shirt', 'FashionIQ-toptee', 
        'MSCOCO', 'Flickr30K',
        'VisDial'])
    
    parser.add_argument('--subset', action='store_true', default=False, help='Use subset of the dataset, for CIRR only')
    
    # Use valid for FashionIQ and VisDial, test for CIRCO, CIRR, MSCOCO and Flickr30K
    parser.add_argument('--split', type=str, choices=["valid", "test"], default="valid")
    
    # The following arguments are for restoring the states from a previous run. Ignore if not needed.
    parser.add_argument('--restore_states', type=str, default=None)
    parser.add_argument('--stage_skip_num', type=int, choices=[0,1,2,3,4], default=0)
    parser.add_argument('--force', action='store_true', default=False)
    args = parser.parse_args()

    xr = xr(args)
    if args.dataset == 'VisDial':
        xr.start_dialog()
    else:
        xr.start()

    