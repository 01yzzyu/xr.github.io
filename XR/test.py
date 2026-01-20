# run_xr.py 🚀
import argparse
from xr import xr

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="External runner for xr",
        add_help=True
    )
    parser.add_argument('--llm_path', type=str, default="/mnt/GeneralModel/yangzhongyu1/CIR/OpenGVLab/InternVL3-8B", help='Reasoner model path')
    parser.add_argument('--mllm_path', type=str, default="/mnt/GeneralModel/yangzhongyu1/CIR/OpenGVLab/InternVL3-8B", help='Verifier model path')
    parser.add_argument('--img_cap_model_path', type=str, default="/mnt/GeneralModel/yangzhongyu1/CIR/OpenGVLab/InternVL3-8B", help='Captioner model path')
    parser.add_argument('--eval_mllm_path', type=str, default="/mnt/GeneralModel/yangzhongyu1/CIR/OpenGVLab/InternVL3-8B", help='Evaluator model path')
    parser.add_argument('--clip_path', type=str, default="/mnt/GeneralModel/yangzhongyu1/CIR/laion/CLIP-ViT-L-14-laion2B-s32B-b82K", help='VLM path')
    parser.add_argument('--dataset_path', type=str, default="/mnt/GeneralModel/yangzhongyu1/CIR/data/", help='Dataset folder path')

    parser.add_argument('--run_name', type=str, default='external')

    parser.add_argument('--tau', type=float, default=0.15)
    parser.add_argument('--top_k', type=int, default=100)
    parser.add_argument('--alpha', type=int, default=100)
    parser.add_argument('--stages', nargs='+', choices=['s1', 's2', 's3'], default=['s1', 's2', 's3'])

    parser.add_argument('--dataset', type=str, choices=[
        'CIRR', 'CIRCO',
        'FashionIQ-dress', 'FashionIQ-shirt', 'FashionIQ-toptee',
        'MSCOCO', 'Flickr30K',
        'VisDial'
    ], required=True)
    parser.add_argument('--subset', action='store_true', default=False, help='Use subset of the dataset (CIRR only)')

    # FashionIQ/VisDial 常用 valid；CIRR/CIRCO/MSCOCO/Flickr30K 可选 valid/test
    parser.add_argument('--split', type=str, choices=["valid", "test"], default="valid")

    # 断点恢复相关
    parser.add_argument('--restore_states', type=str, default=None)
    parser.add_argument('--stage_skip_num', type=int, choices=[0, 1, 2, 3, 4], default=0)
    parser.add_argument('--force', action='store_true', default=False)
    return parser

def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    runner = xr(args)
    # VisDial 走多轮对话入口
    if args.dataset == 'VisDial':
        runner.start_dialog()
    else:
        runner.start()

if __name__ == "__main__":
    main()
