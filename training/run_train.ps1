# 复现训练（Windows PowerShell）
# 用法：.\run_train.ps1
# 前提：pip install torch transformers peft

$env:PYTHONUTF8 = "1"
$env:BASE_MODEL = "Qwen/Qwen3-0.6B"                     # 或本地路径
$env:TRAIN_FILE = "data/train_mix.jsonl"
$env:DEV_FILE   = "data/dev_mix.jsonl"
$env:TRAIN_OUT  = "out/lora_router"
$env:BATCH      = "4"

python train_judge.py
