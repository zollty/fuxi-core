
def add_model_args(parser):
    parser.add_argument(
        "--model-path",
        type=str,
        default="lmsys/vicuna-7b-v1.5",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="Hugging Face Hub model revision identifier",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps", "xpu", "npu"],
        default="cuda",
        help="The device type",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="The maximum memory per GPU for storing model weights. Use a string like '13Gib'",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--load-8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--cpu-offloading",
        action="store_true",
        help="Only when using 8-bit quantization: Offload excess weights to the CPU that don't fit on the GPU",
    )
    parser.add_argument(
        "--gptq-ckpt",
        type=str,
        default=None,
        help="Used for GPTQ. The path to the local GPTQ checkpoint.",
    )
    parser.add_argument(
        "--gptq-wbits",
        type=int,
        default=16,
        choices=[2, 3, 4, 8, 16],
        help="Used for GPTQ. #bits to use for quantization",
    )
    parser.add_argument(
        "--gptq-groupsize",
        type=int,
        default=-1,
        help="Used for GPTQ. Groupsize to use for quantization; default uses full row.",
    )
    parser.add_argument(
        "--gptq-act-order",
        action="store_true",
        help="Used for GPTQ. Whether to apply the activation order GPTQ heuristic",
    )
    parser.add_argument(
        "--awq-ckpt",
        type=str,
        default=None,
        help="Used for AWQ. Load quantized model. The path to the local AWQ checkpoint.",
    )
    parser.add_argument(
        "--awq-wbits",
        type=int,
        default=16,
        choices=[4, 16],
        help="Used for AWQ. #bits to use for AWQ quantization",
    )
    parser.add_argument(
        "--awq-groupsize",
        type=int,
        default=-1,
        help="Used for AWQ. Groupsize to use for AWQ quantization; default uses full row.",
    )
    parser.add_argument(
        "--enable-exllama",
        action="store_true",
        help="Used for exllamabv2. Enable exllamaV2 inference framework.",
    )
    parser.add_argument(
        "--exllama-max-seq-len",
        type=int,
        default=4096,
        help="Used for exllamabv2. Max sequence length to use for exllamav2 framework; default 4096 sequence length.",
    )
    parser.add_argument(
        "--exllama-gpu-split",
        type=str,
        default=None,
        help="Used for exllamabv2. Comma-separated list of VRAM (in GB) to use per GPU. Example: 20,7,7",
    )
    parser.add_argument(
        "--exllama-cache-8bit",
        action="store_true",
        help="Used for exllamabv2. Use 8-bit cache to save VRAM.",
    )
    parser.add_argument(
        "--enable-xft",
        action="store_true",
        help="Used for xFasterTransformer Enable xFasterTransformer inference framework.",
    )
    parser.add_argument(
        "--xft-max-seq-len",
        type=int,
        default=4096,
        help="Used for xFasterTransformer. Max sequence length to use for xFasterTransformer framework; default 4096 sequence length.",
    )
    parser.add_argument(
        "--xft-dtype",
        type=str,
        choices=["fp16", "bf16", "int8", "bf16_fp16", "bf16_int8"],
        help="Override the default dtype. If not set, it will use bfloat16 for first token and float16 next tokens on CPU.",
        default=None,
    )
