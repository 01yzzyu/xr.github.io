import os
import time
import random
from copy import deepcopy
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
import ray

from vllm import LLM, SamplingParams
from transformers import (
    CLIPModel,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,  # kept for compatibility even if unused
)

from utils.data import ImageFeatureDataset
from utils.function import (
    resize_and_concatenate,
    concatenate_images_with_reference,
    resize_image_ratio,
)

# -----------------------------------------------------------------------------
# 全局保守设置：避免多 worker 线程争抢导致抖动
# -----------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
torch.set_num_threads(1)

# 固定随机种子（可按需去掉）
SEED = int(os.environ.get("xr_SEED", "42"))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# =========================
# Utilities
# =========================
def _resolve_model_path(path: str) -> str:
    """
    Resolve model path with default-fallback logic:
    1) If user provided a non-empty path, use it as-is.
    2) Else use env DEFAULT_MLLM_PATH (if set).
    3) Else use env INTERNVL3_MODEL_PATH (if set).
    4) Else fallback to 'OpenGVLab/InternVL3-8B'.
    """
    if path is not None and str(path).strip() != "":
        return str(path).strip()
    env1 = os.getenv("DEFAULT_MLLM_PATH", "").strip()
    if env1:
        return env1
    env2 = os.getenv("INTERNVL3_MODEL_PATH", "").strip()
    if env2:
        return env2
    return "OpenGVLab/InternVL3-8B"


def _auto_processor_with_fast(model_path):
    """
    优先尝试使用 use_fast=True 的处理器（transformers >= 4.40 一些模型才支持）。
    对不支持 use_fast 的版本做降级。
    返回值与原来保持一致：
      - 对 CLIP：通常 AutoProcessor 返回 CLIPProcessor，包含 .tokenizer / .image_processor
    """
    try:
        proc = AutoProcessor.from_pretrained(model_path, use_fast=True)
        return proc
    except TypeError:
        proc = AutoProcessor.from_pretrained(model_path)
        return proc


def _to_cuda_float_tensor_from_numpy(arr: np.ndarray) -> torch.Tensor:
    """
    把 numpy 数组安全转换为 GPU float tensor：
    - 强制 copy + ascontiguous，避免只读/非连续导致的 PyTorch 警告与潜在不确定行为
    """
    arr_safe = np.ascontiguousarray(np.array(arr, copy=True))
    return torch.from_numpy(arr_safe).to("cuda").float()


def _sanitize_text_input(text: str) -> str:
    """
    避免文本中出现字面量 "<image>" 被 tokenizer 当成特殊占位符计数。
    """
    return text.replace("<image>", "[IMAGE]").replace("<IMAGE>", "[IMAGE]")


# ---------- 多模态占位符一致性检查工具 ----------
def _count_image_placeholders(tokenizer, prompt_str: str) -> int:
    """
    通过 tokenizer 编码后统计 <image> token 数量。
    若无法可靠获取 special id，则退回字符串计数。
    """
    try:
        image_tok_id = tokenizer.convert_tokens_to_ids("<image>")
        if image_tok_id is None or (isinstance(image_tok_id, int) and image_tok_id < 0):
            return prompt_str.count("<image>")
        ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        return sum(1 for i in ids if i == image_tok_id)
    except Exception:
        return prompt_str.count("<image>")


def _images_in_multi_modal(mm_dict) -> int:
    """
    统计 multi_modal_data 中的图像数量：
      - {"image": PIL.Image}
      - {"image": [PIL.Image, ...]}
    """
    if not isinstance(mm_dict, dict):
        return 0
    imgs = mm_dict.get("image", None)
    if imgs is None:
        return 0
    if isinstance(imgs, (list, tuple)):
        return len(imgs)
    return 1  # 单图对象


def _validate_mm_inputs_or_raise(tokenizer, inputs_list):
    """
    批量一致性校验：
    - prompt 中 <image> 次数 == multi_modal_data 中的图片数量
    - 不一致则抛出 ValueError（便于快速定位数据问题）
    """
    for idx, sample in enumerate(inputs_list):
        prompt_str = sample.get("prompt", "")
        mm = sample.get("multi_modal_data", {}) or {}
        n_ph = _count_image_placeholders(tokenizer, prompt_str)
        n_img = _images_in_multi_modal(mm)
        if n_ph != n_img:
            short_preview = prompt_str[:300].replace("\n", "\\n")
            raise ValueError(
                f"[MM-VALIDATION] Sample #{idx}: prompt expects {n_ph} image(s) "
                f"but {n_img} provided. First-300-chars: `{short_preview}...`"
            )


# =========================
# LLM: text-only
# =========================
@ray.remote
def llm_load_and_inference_ray(llm_path, input_prompt):
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048)
    effective_path = _resolve_model_path(llm_path)
    model = LLM(
        model=effective_path,
        trust_remote_code=True,
        max_model_len=4096,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        enable_prefix_caching=True,
    )
    tokenizer = model.get_tokenizer()
    _sta_time = time.time()
    conv_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in input_prompt
    ]
    llm_outputs = model.generate([{"prompt": p} for p in conv_prompts], sampling_params)
    _end_time = time.time()
    _time_interval = _end_time - _sta_time
    info_dict = {"time": _time_interval, "num_task": len(input_prompt)}
    return (llm_outputs, info_dict)


# =========================
# MLLM: general batch inference
# =========================
@ray.remote
def mllm_load_and_inference_ray(mllm_path, mllm_query):
    mllm_outputs = defaultdict(list)
    mini_batch_size = 16000
    stop_strings = None
    stop_token_ids = None
    max_model_len = None

    effective_path = _resolve_model_path(mllm_path)
    ep_low = (effective_path or "").lower()

    # ===== prompt template by model family =====
    # InternVL/默认：使用“块列表”多模态消息（image+text）
    use_block_messages = ("internvl" in ep_low)

    # Qwen2.5-VL / Qwen-VL 系列关键字集合（尽量覆盖常见写法）
    _is_qwen_vl = any(s in ep_low for s in [
        "qwen2.5-vl", "qwen2.5vl", "qwen2vl", "qwen-vl", "qwenvl"
    ])

    if "llava" in ep_low:
        prompt = "USER: <image>\n{}\nASSISTANT:"
        stop_strings = None
        use_block_messages = False
    elif "paligemma" in ep_low:
        prompt = "{}"
        stop_strings = None
        use_block_messages = False
    elif "phi-3" in ep_low:
        prompt = "<|user|>\n<|image_1|>\n{}<|end|>\n<|assistant|>\n"
        stop_strings = None
        use_block_messages = False
    elif "minicpm" in ep_low:
        # MiniCPM 多为字符串模板；保留字符串形式
        prompt = [{"role": "user", "content": "(<image>./</image>)\n{}"}]
        max_model_len = 4096
        use_block_messages = False
    elif _is_qwen_vl:
        # Qwen2.5-VL 系列：推荐块列表（image+text），由 tokenizer.apply_chat_template 生成
        use_block_messages = True
        stop_strings = None
        # 可按需限制上下文长度（通常无需强制设置）
        # max_model_len = 32768
    else:
        # 默认按 InternVL 风格（块列表）
        use_block_messages = True

    # ===== create model =====
    if "phi-3" in ep_low:
        model = LLM(
            model=effective_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.8,
            max_num_seqs=5,
        )
    else:
        model = LLM(
            model=effective_path,
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            max_model_len=max_model_len,
            trust_remote_code=True,
        )
    tokenizer = model.get_tokenizer()

    # ===== stop token ids only for MiniCPM (if present) =====
    if "minicpm" in ep_low:
        stop_token_ids = [getattr(tokenizer, "eos_id", None)]
        stop_token_ids = [i for i in stop_token_ids if isinstance(i, int)]
        if len(stop_token_ids) == 0:
            stop_token_ids = None

    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=2048,
        stop=stop_strings,            # string stops are robust
        stop_token_ids=stop_token_ids # for MiniCPM if available
    )

    mllm_inputs = defaultdict(list)
    item_idx_list = list(mllm_query.keys())
    total_num = len(item_idx_list)
    current_query_num = 0
    processed_num = 0
    _num_input_token = []
    _num_output_token = []

    _sta_time = time.time()
    for item_idx in item_idx_list:
        text_inputs = mllm_query[item_idx]["text_inputs"]
        if mllm_query[item_idx]["ref_img_path"] is None:
            ref_image_obj = Image.new("RGB", (224, 224))
        else:
            ref_image_obj = Image.open(mllm_query[item_idx]["ref_img_path"]).convert("RGB")

        top_k_ranked_candidates = mllm_query[item_idx]["top_k_ranked_candidates"]
        for candidate_idx in top_k_ranked_candidates.keys():
            candidate_image_path = top_k_ranked_candidates[candidate_idx]["image_path"]
            candidate_image_obj = Image.open(candidate_image_path).convert("RGB")
            # 不手动 resize，交给模型/processor

            for proposition_idx, _text in enumerate(text_inputs):
                text_input = _sanitize_text_input(_text)

                if use_block_messages:
                    # InternVL / Qwen2.5-VL 等：以块列表表示多模态
                    # 注意：这里 image 占位即可，图像数据通过 multi_modal_data 传入
                    message = [{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text_input},
                        ],
                    }]
                    item_prompt = tokenizer.apply_chat_template(
                        message, tokenize=False, add_generation_prompt=True
                    )
                else:
                    # 其它模型保持原字符串模板（如 LLaVA / Phi-3 / MiniCPM）
                    if isinstance(prompt, str):
                        item_prompt = prompt.format(text_input)
                    else:
                        message = deepcopy(prompt)
                        message[0]["content"] = prompt[0]["content"].format(text_input)
                        item_prompt = tokenizer.apply_chat_template(
                            message, tokenize=False, add_generation_prompt=True
                        )

                mllm_inputs["index"].append((item_idx, candidate_idx, proposition_idx))
                mllm_inputs["inputs"].append(
                    {
                        "prompt": item_prompt,
                        "multi_modal_data": {
                            # vLLM 多模态约定：键 "image" 与上面的 {"type":"image"} 顺序/数量对齐
                            "image": candidate_image_obj
                        }
                    }
                )

        current_query_num += 1
        if len(mllm_inputs["index"]) >= mini_batch_size or item_idx == item_idx_list[-1]:
            print(
                "---MLLM Batch Inference from {} to {} , total {} ---".format(
                    processed_num, processed_num + current_query_num, total_num
                )
            )
            processed_num += current_query_num
            current_query_num = 0

            # 一致性校验（确保占位与 multi_modal_data 对齐）
            _validate_mm_inputs_or_raise(tokenizer, mllm_inputs["inputs"])

            outputs = model.generate(mllm_inputs["inputs"], sampling_params)
            for (item_idx, candidate_idx, proposition_idx), o in zip(
                mllm_inputs["index"], outputs
            ):
                generated_text = o.outputs[0].text
                _num_input_token.append(len(o.prompt_token_ids))
                _num_output_token.append(len(o.outputs[0].token_ids))
                if item_idx not in mllm_outputs.keys():
                    mllm_outputs[item_idx] = defaultdict(list)
                mllm_outputs[item_idx][candidate_idx].append(generated_text)
            mllm_inputs.clear()

    _end_time = time.time()
    _time_interval = _end_time - _sta_time
    info_dict = {
        "time": _time_interval,
        "num_task": len(_num_input_token),
        "num_input_token": _num_input_token,
        "num_output_token": _num_output_token,
    }

    return (mllm_outputs, info_dict)

# =========================
# Stage3 QA (turn-based)
# =========================
@ray.remote
def stage3_large_mllm_inference_qa_ray(mllm_path, mllm_query):
    mllm_outputs = defaultdict(list)
    sampling_params = SamplingParams(
        temperature=0, top_p=1, max_tokens=256, stop=["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    )

    effective_path = _resolve_model_path(mllm_path)
    ep_low = (effective_path or "").lower()
    model = LLM(
        model=effective_path,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    # InternVL → 用块列表；其它保持字符串模板
    use_block_messages = ("internvl" in ep_low)

    if "llava-v1.6" in ep_low:
        prompt = "[INST] <image>\n{} [/INST]"
        use_block_messages = False
    elif "paligemma" in ep_low:
        prompt = "{}"
        use_block_messages = False
    elif "phi-3" in ep_low:
        prompt = "<|user|>\n<|image_1|>\n{}<|end|>\n<|assistant|>\n"
        use_block_messages = False
    elif "minicpm" in ep_low:
        prompt = [{"role": "user", "content": "(<image>./</image>)\n{}"}]
        use_block_messages = False

    _total_time = 0
    _total_task = 0
    _keys = list(mllm_query.keys())
    mllm_query_flag = {k: False for k in mllm_query.keys()}
    info_dict = dict()
    tokenizer = model.get_tokenizer()

    turns = len(mllm_query[_keys[0]]["top_candidate"]) if _keys else 0

    for turn_idx in range(turns):
        print(f"Turn {turn_idx}")
        _sta_time = time.time()
        mllm_turn_inputs = {"index": list(), "inputs": list()}
        for item_idx in mllm_query.keys():
            if mllm_query_flag[item_idx]:
                continue
            reference_img_path = mllm_query[item_idx]["ref_img_path"]
            if reference_img_path is None:
                reference_img_obj = Image.new("RGB", (224, 224))
            else:
                reference_img_obj = Image.open(reference_img_path).convert("RGB")
            candidate_img_path = mllm_query[item_idx]["top_candidate"][turn_idx]["image_path"]
            candidate_img_obj = Image.open(candidate_img_path).convert("RGB")
            text_input = _sanitize_text_input(mllm_query[item_idx]["text_input"])

            if use_block_messages:
                message = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": text_input},
                    ],
                }]
                built_prompt = tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
            else:
                if isinstance(prompt, str):
                    built_prompt = prompt.format(text_input)
                else:
                    message = deepcopy(prompt)
                    message[0]["content"] = prompt[0]["content"].format(text_input)
                    built_prompt = tokenizer.apply_chat_template(
                        message, tokenize=False, add_generation_prompt=True
                    )

            mllm_turn_inputs["index"].append((item_idx, turn_idx))
            mllm_turn_inputs["inputs"].append(
                {"prompt": built_prompt, "multi_modal_data": {"image": candidate_img_obj}}
            )
        _total_task += len(mllm_turn_inputs["index"])

        _validate_mm_inputs_or_raise(tokenizer, mllm_turn_inputs["inputs"])

        outputs = model.generate(mllm_turn_inputs["inputs"], sampling_params)

        for (item_idx, turn_idx), o in zip(mllm_turn_inputs["index"], outputs):
            generated_text = o.outputs[0].text
            decision = False
            if generated_text.lower().strip().startswith("yes"):
                mllm_query_flag[item_idx] = True
                decision = True

            if item_idx <= 10:
                print(f"{item_idx}, {turn_idx}\n {generated_text}")

            mllm_outputs[item_idx].append(
                {
                    "turn": turn_idx,
                    "decision": decision,
                    "text": generated_text,
                    "num_input_token": len(o.prompt_token_ids),
                    "num_output_token": len(o.outputs[0].token_ids),
                }
            )

        _end_time = time.time()
        _turn_time_interval = _end_time - _sta_time
        _total_time += _turn_time_interval
        info_dict[f"turn_{turn_idx}_time"] = _turn_time_interval

        if all(mllm_query_flag.values()):
            break

    info_dict = {"time": _total_time, "num_task": _total_task}
    return (mllm_outputs, info_dict)


# =========================
# Stage3 pairwise-batch comparison (robust)
# =========================
@ray.remote
def stage3_large_mllm_inference_batch_cmp_ray(mllm_path, mllm_query):
    """
    Batch pairwise-eval for stage3 with robust handling:
      - context-length truncation (safe upper bound)
      - safe stop strings
      - unified return format: (result_dict, stats_dict)
      - guards for empty/partial inputs to avoid IndexError/KeyError
      - resilient image loading
      - InternVL：两图输入 -> 两个 image 块 + 两张图
    """
    import time
    from collections import defaultdict
    from copy import deepcopy
    from PIL import Image
    from vllm import LLM, SamplingParams

    def _resolve(safe_dict, key, default=None):
        return safe_dict.get(key, default) if isinstance(safe_dict, dict) else default

    def _safe_open_rgb(path, size=None):
        """尽量打开图片；size=None 表示不 resize。"""
        try:
            if path:
                img = Image.open(path).convert("RGB")
                if size:
                    img = img.resize(size)
                return img
        except Exception:
            pass
        return Image.new("RGB", size or (224, 224), color=(128, 128, 128))

    effective_path = _resolve_model_path(mllm_path)
    ep_low = (effective_path or "").lower()

    # Prompt & settings
    stop_strings = None
    other_args = {}
    use_block_messages = ("internvl" in ep_low)
    two_images = ("internvl" in ep_low)  # 仅 InternVL 用双图；其它模型维持单图

    if "llava-v1.6-mistral" in ep_low:
        prompt_template = "[INST] <image>\n{} [/INST]"
        use_block_messages = False
        two_images = False
    elif "llava-v1.6-vicuna" in ep_low:
        prompt_template = "USER: <image>\n{}\nASSISTANT:"
        use_block_messages = False
        two_images = False
    elif "llava-v1.6-34b" in ep_low:
        prompt_template = "<|im_start|>user\n<image>\n{}<|im_end|>\n"
        use_block_messages = False
        two_images = False
        other_args = {"quantization": "fp8", "trust_remote_code": True}
    else:
        # InternVL 或默认
        if use_block_messages:
            prompt_template = None  # 块列表渲染，不用字符串模板
            stop_strings = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
        else:
            prompt_template = "USER: <image>\n{}\nASSISTANT:"

    model = LLM(
        model=effective_path,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        **other_args,
    )
    tokenizer = model.get_tokenizer()

    # 取安全的最大上下文（保守）
    def _derived_max_len(tok):
        val = getattr(tok, "model_max_length", None)
        if isinstance(val, int) and 0 < val < 10_000_000:
            return val
        cfg = getattr(tok, "init_kwargs", {}) or {}
        if isinstance(cfg.get("model_max_length"), int):
            return cfg["model_max_length"]
        m = getattr(tok, "model", None)
        mpe = getattr(getattr(m, "config", None), "max_position_embeddings", None) if m else None
        return int(mpe) if isinstance(mpe, int) and mpe > 0 else 12288

    DERIVED_MAX = _derived_max_len(tokenizer)
    GEN_MAX_TOKENS = 256
    SAFETY_MARGIN = 64

    def _truncate_to_ctx(prompt_str: str) -> str:
        try:
            ids = tokenizer.encode(prompt_str, add_special_tokens=False)
        except Exception:
            return prompt_str
        allowed = max(512, DERIVED_MAX - GEN_MAX_TOKENS - SAFETY_MARGIN)
        if len(ids) <= allowed:
            return prompt_str
        ids = ids[-allowed:]
        try:
            return tokenizer.decode(ids, skip_special_tokens=False)
        except Exception:
            return prompt_str

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=GEN_MAX_TOKENS,
        stop=stop_strings,
        repetition_penalty=1.0,
    )

    _total_time = 0.0
    _total_task = 0
    mllm_outputs = defaultdict(list)
    info_dict = {"turn_time": [], "turn_task": []}

    if not isinstance(mllm_query, dict) or not mllm_query:
        info_dict["time"] = 0.0
        info_dict["num_task"] = 0
        return (dict(mllm_outputs), info_dict)

    _keys = list(mllm_query.keys())
    first_key = _keys[0]
    first_obj = mllm_query.get(first_key, {})
    top_candidates = first_obj.get("top_candidate", []) or first_obj.get("top_k_ranked_candidates", {})
    if isinstance(top_candidates, dict):
        turns = len(top_candidates)
        def _get_turn(obj, idx): return obj.get(idx, {})
    else:
        turns = len(top_candidates)
        def _get_turn(obj, idx):
            try: return obj[idx]
            except Exception: return {}

    mllm_query_flag = {k: False for k in _keys}

    for turn_idx in range(max(0, turns)):
        _sta_time = time.time()
        batch_indices = []
        batch_inputs = []

        for item_idx in _keys:
            if mllm_query_flag.get(item_idx, False):
                continue

            item = mllm_query.get(item_idx, {})
            ref_path = item.get("ref_img_path", None)
            reference_img_obj = _safe_open_rgb(ref_path, size=None)       # 不 resize
            tc = item.get("top_candidate", None)
            if tc is None:
                tc = item.get("top_k_ranked_candidates", None)
            cand_meta = _get_turn(tc, turn_idx) if tc is not None else {}
            cand_path = (cand_meta or {}).get("image_path", None)
            candidate_img_obj = _safe_open_rgb(cand_path, size=None)      # 不 resize

            text_input_raw = item.get("text_input", "") or item.get("text_inputs", "")
            if isinstance(text_input_raw, list):
                text_input_raw = text_input_raw[turn_idx] if 0 <= turn_idx < len(text_input_raw) else (text_input_raw[-1] if text_input_raw else "")
            text_input = _sanitize_text_input(text_input_raw)

            if use_block_messages:
                if two_images:
                    # InternVL：两图 -> 两个 image 块 + 传两张图
                    message = [{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": text_input},
                        ],
                    }]
                    prompt_str = tokenizer.apply_chat_template(
                        message, tokenize=False, add_generation_prompt=True
                    )
                    images_to_use = [reference_img_obj, candidate_img_obj]
                else:
                    message = [{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text_input},
                        ],
                    }]
                    prompt_str = tokenizer.apply_chat_template(
                        message, tokenize=False, add_generation_prompt=True
                    )
                    images_to_use = candidate_img_obj
            else:
                prompt_str = prompt_template.format(text_input)
                images_to_use = candidate_img_obj

            prompt_str = _truncate_to_ctx(prompt_str)

            batch_indices.append((item_idx, turn_idx))
            batch_inputs.append({"prompt": prompt_str, "multi_modal_data": {"image": images_to_use}})

        _total_task += len(batch_indices)
        if not batch_indices:
            info_dict["turn_time"].append(0.0)
            info_dict["turn_task"].append(0)
            break

        _validate_mm_inputs_or_raise(tokenizer, batch_inputs)

        outputs = model.generate(batch_inputs, sampling_params)

        for (itm, t_idx), o in zip(batch_indices, outputs):
            try:
                out = o.outputs[0]
                generated_text = getattr(out, "text", "") or ""
                gg = generated_text.strip().lower()
                decision = gg.startswith("yes") or ("answer: yes" in gg)

                if decision:
                    mllm_query_flag[itm] = True

                mllm_outputs[itm].append(
                    {
                        "turn": t_idx,
                        "decision": bool(decision),
                        "text": generated_text,
                        "num_input_token": len(getattr(o, "prompt_token_ids", []) or []),
                        "num_output_token": len(getattr(out, "token_ids", []) or []),
                    }
                )
            except Exception:
                mllm_outputs[itm].append(
                    {
                        "turn": t_idx,
                        "decision": False,
                        "text": "",
                        "num_input_token": 0,
                        "num_output_token": 0,
                    }
                )

        _turn_time = time.time() - _sta_time
        _total_time += _turn_time
        info_dict["turn_time"].append(_turn_time)
        info_dict["turn_task"].append(len(batch_indices))

        if all(mllm_query_flag.values()):
            break

    info_dict["time"] = _total_time
    info_dict["num_task"] = _total_task

    return (dict(mllm_outputs), info_dict)


# =========================
# Captioning
# =========================
@ray.remote
def mllm_inference_for_caption_ray(mllm_path, mllm_query):
    mllm_outputs = defaultdict(list)
    sampling_params = SamplingParams(
        temperature=0,
        top_p=1,
        max_tokens=256,
        repetition_penalty=1.05,
        stop=["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"],
    )
    effective_path = _resolve_model_path(mllm_path)
    ep_low = (effective_path or "").lower()
    model = LLM(
        model=effective_path,
        enforce_eager=True,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )
    tokenizer = model.get_tokenizer()
    use_block_messages = ("internvl" in ep_low)

    # 仅当不是 InternVL 时，才用旧字符串模板
    prompt_str_tpl = "USER:<image>\n{}\nASSISTANT:"

    mini_batch_size = 10000
    keys = list(mllm_query.keys())
    batch_idx = 0
    mllm_batch_inputs = []
    mllm_batch_keys = []

    for key in keys:
        mllm_batch_keys.append(key)
        image_obj = Image.open(mllm_query[key]["image_path"]).convert("RGB")
        text_input = _sanitize_text_input(mllm_query[key]["text_input"])

        if use_block_messages:
            message = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_input},
                ],
            }]
            built_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        else:
            built_prompt = prompt_str_tpl.format(text_input)

        mllm_batch_inputs.append({"prompt": built_prompt, "multi_modal_data": {"image": image_obj}})

        if len(mllm_batch_inputs) >= mini_batch_size or key == keys[-1]:
            sta = batch_idx * mini_batch_size
            end = sta + len(mllm_batch_inputs)
            print("---MLLM Batch Inference from {} to {} , total {} ---".format(sta, end, len(keys)))

            _validate_mm_inputs_or_raise(tokenizer, mllm_batch_inputs)

            outputs = model.generate(mllm_batch_inputs, sampling_params)
            for k, o in zip(mllm_batch_keys, outputs):
                generated_text = o.outputs[0].text
                mllm_outputs[k] = generated_text
            batch_idx += 1
            mllm_batch_inputs = []
            mllm_batch_keys = []
    return mllm_outputs


# =========================
# CLIP feature extraction
# =========================
@ray.remote
def clip_extact_feature_ray(clip_path, image_files, image_caption):
    model = CLIPModel.from_pretrained(clip_path)
    # 优先 fast 处理器
    try:
        processor = _auto_processor_with_fast(clip_path).image_processor
    except Exception:
        processor = AutoProcessor.from_pretrained(clip_path).image_processor
    tokenizer = AutoTokenizer.from_pretrained(clip_path)

    dataset = ImageFeatureDataset(image_files, image_caption, processor, tokenizer)
    image_dataloader = DataLoader(
        dataset,
        batch_size=512,
        collate_fn=dataset.collator,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    model = model.eval().to("cuda")
    image_embeddings = []
    caption_embeddings = []
    image_name_list = []

    with torch.no_grad():
        for batch in tqdm(image_dataloader, desc="Inference"):
            image_names = batch.pop("images_name")
            pixel_values = batch.pop("images_tensor").cuda(non_blocking=True)
            image_embedding = model.get_image_features(pixel_values)
            image_embedding = image_embedding.detach().cpu().numpy()
            image_embeddings.append(image_embedding)

            for key in batch.keys():
                batch[key] = batch[key].cuda(non_blocking=True)
            caption_embedding = model.get_text_features(**batch)
            caption_embedding = caption_embedding.detach().cpu().numpy()
            caption_embeddings.append(caption_embedding)

            image_name_list.extend(image_names)

    image_embeddings = np.concatenate(image_embeddings, axis=0)
    caption_embeddings = np.concatenate(caption_embeddings, axis=0)
    return_dict = {
        "image_embedding": image_embeddings,
        "caption_embedding": caption_embeddings,
        "image_name_list": image_name_list,
    }
    return return_dict


# =========================
# CLIP ranking
# =========================
@ray.remote
def clip_rank_retrieval_ray(args, query_states, image_db_embedding, image_caption_embedidng, image_db_index):
    # 模型与处理器
    clip_model = CLIPModel.from_pretrained(args.clip_path).to("cuda").eval()
    try:
        processor = _auto_processor_with_fast(args.clip_path)
    except Exception:
        processor = AutoProcessor.from_pretrained(args.clip_path)

    # 索引与向量（安全转换，消除只读/非连续警告）
    image_db_index = image_db_index.tolist() if hasattr(image_db_index, "tolist") else list(image_db_index)
    image_db_embedding = _to_cuda_float_tensor_from_numpy(image_db_embedding)
    image_db_embedding = F.normalize(image_db_embedding, dim=1)
    image_caption_embedidng = _to_cuda_float_tensor_from_numpy(image_caption_embedidng)
    image_caption_embedidng = F.normalize(image_caption_embedidng, dim=1)

    clip_rank_results = defaultdict(dict)
    mini_batch_size = 128
    inference_data = []

    _sta_time = time.time()
    with torch.no_grad():
        query_keys = list(query_states.keys())
        for item_idx in tqdm(query_keys, total=len(query_keys), desc="[Stage1] CLIP Ranking"):
            tgt_img_desc_list = query_states[item_idx]["result"]["stage1_reason"]["json"]["step2"]

            ref_img = query_states[item_idx]["info"]["ref_img"]
            if ref_img is None:
                ref_img_obj = Image.new("RGB", (224, 224))
            else:
                ref_img_obj = Image.open(query_states[item_idx]["info"]["ref_img_path"])

            if len(tgt_img_desc_list):
                for tgt_img_desc in tgt_img_desc_list:
                    inference_data.append(
                        {
                            "item_idx": item_idx,
                            "num_of_tgt_img_desc": len(tgt_img_desc_list),
                            "tgt_img_desc": tgt_img_desc,
                            "ref_img": ref_img,
                            "ref_img_obj": ref_img_obj,
                        }
                    )
            else:
                inference_data.append(
                    {
                        "item_idx": item_idx,
                        "num_of_tgt_img_desc": 1,
                        "tgt_img_desc": query_states[item_idx]["info"]["inst"],
                        "ref_img": ref_img,
                        "ref_img_obj": ref_img_obj,
                    }
                )

            if len(inference_data) >= mini_batch_size or item_idx == query_keys[-1]:
                batch_inference_data = inference_data
                inference_data = []
                batch_item_ids = [item["item_idx"] for item in batch_inference_data]
                batch_num_of_tgt_img_desc = [item["num_of_tgt_img_desc"] for item in batch_inference_data]
                batch_tgt_img_desc = [item["tgt_img_desc"] for item in batch_inference_data]
                batch_ref_imgs = [item["ref_img"] for item in batch_inference_data]
                batch_ref_img_objs = [item["ref_img_obj"] for item in batch_inference_data]

                # 文本特征
                text_inputs = processor.tokenizer(
                    batch_tgt_img_desc, return_tensors="pt", padding=True, truncation=True
                )
                text_inputs = {k: v.to("cuda") for k, v in text_inputs.items()}
                text_embeddings = clip_model.get_text_features(**text_inputs)
                text_embeddings = F.normalize(text_embeddings, dim=1).float()

                # 参考图像（用于一致流程；不参与融合也可）
                image_inputs = processor.image_processor(batch_ref_img_objs, return_tensors="pt")[
                    "pixel_values"
                ].to("cuda")
                image_embeddings = clip_model.get_image_features(pixel_values=image_inputs)
                image_embeddings = F.normalize(image_embeddings, dim=1).float()
                _ = image_embeddings  # placeholder

                # 检索融合
                text_similarity = torch.matmul(text_embeddings, image_db_embedding.T)
                image_similarity = torch.matmul(text_embeddings, image_caption_embedidng.T)
                similarity = text_similarity * (1 - args.tau) + image_similarity * args.tau
                similarity = similarity.cpu()

                for batch_idx, (item_idx, num_of_tgt_img_desc) in enumerate(
                    zip(batch_item_ids, batch_num_of_tgt_img_desc)
                ):
                    if "similarity" not in clip_rank_results[item_idx].keys():
                        clip_rank_results[item_idx]["similarity"] = similarity[batch_idx]
                    else:
                        clip_rank_results[item_idx]["similarity"] += similarity[batch_idx]
                    clip_rank_results[item_idx]["num_of_tgt_img_desc"] = num_of_tgt_img_desc

        for item_idx in tqdm(clip_rank_results.keys()):
            num_of_tgt_img_desc = clip_rank_results[item_idx].pop("num_of_tgt_img_desc")
            clip_rank_results[item_idx]["similarity"] /= max(num_of_tgt_img_desc, 1)
            clip_rank_results[item_idx]["sorted_indices"] = torch.argsort(
                clip_rank_results[item_idx]["similarity"], descending=True
            ).tolist()
            clip_rank_results[item_idx]["similarity"] = (
                clip_rank_results[item_idx]["similarity"][clip_rank_results[item_idx]["sorted_indices"]].tolist()
            )
            clip_rank_results[item_idx]["sorted_names"] = [
                image_db_index[i] for i in clip_rank_results[item_idx]["sorted_indices"]
            ]

    _end_time = time.time()
    _time_interval = _end_time - _sta_time
    return (clip_rank_results, {"time": _time_interval, "num_task": len(query_keys)})
