"""最小推理示例：加载 adapter，把一句话路由成一条 JSON。

用法：
    python examples/infer.py                      # 跑 data 里的第一条样本做演示
    python examples/infer.py --task "把模型切到 qwen2.5-0.5b"
    python examples/infer.py --eval               # 跑 dev_eval_60，报精确匹配

依赖：
    pip install torch transformers peft
"""

import argparse
import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ADAPTER = os.path.join(ROOT, "adapter")
DATA = os.path.join(ROOT, "training", "data")
BASE = os.environ.get("BASE_MODEL") or "Qwen/Qwen3-0.6B"

# 必须和训练时逐字一致（见 training/train_judge.py 的 SYSTEM 常量）
SYSTEM = (
    '你是电脑协调器。你只有下面列出的技能，不要发明技能。\n'
    '只输出一行 JSON，不要解释，不要复述任务。\n'
    '先判断：这条任务里有没有靠列出的技能能做完的部分。\n'
    '三种输出：\n'
    '能做完：{"action":"call","skill":"技能名","params":{...}}\n'
    '只能做完一部分：{"action":"call","skill":"技能名","params":{...},"note":"剩下的做不了的原因"}\n'
    '整体做不了——缺信息、问一句就能做：{"action":"ask","need":"要问什么"}\n'
    '整体做不了——压根没有这个能力：{"action":"refuse","reason":"短原因"}\n'
    '判断标准：任务里有多个诉求时，先做能用技能做的那部分，'
    '不要因为附带的小要求就整条拒绝。\n'
    '任务必须和技能描述的事情直接相关才能 call，绝对不能为了有输出就硬塞技能。\n'
    '缺关键信息（没给路径、没说哪个模型）就 ask，不要猜。\n'
)


def load():
    tok = AutoTokenizer.from_pretrained(ADAPTER)
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    return tok, model


def user_text(manifest, task):
    # 注意：这个拼接格式必须和训练时一模一样
    return f'{manifest}\n任务：{task}\n输出 JSON：'


@torch.no_grad()
def route(tok, model, manifest, task):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user_text(manifest, task)},
    ]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        max_new_tokens=120,
        do_sample=False,          # 贪心解码，和评测口径一致
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def norm(s):
    # 和训练脚本 evaluate() 完全一致：去掉所有空格 + 全角引号归一
    return s.replace(" ", "").replace("“", '"').replace("”", '"').strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", help="直接给一句话（会用第一条样本的 manifest）")
    ap.add_argument("--eval", action="store_true", help="跑 dev_eval_60.jsonl 报分")
    args = ap.parse_args()

    tok, model = load()
    demo = read_jsonl(os.path.join(DATA, "dev_eval_60.jsonl"))[0]

    if args.eval:
        rows = read_jsonl(os.path.join(DATA, "dev_eval_60.jsonl"))
        ok, bad = 0, []
        for i, r in enumerate(rows, 1):
            got = route(tok, model, r["manifest"], r["task"])
            if norm(got) == norm(r["output"]):
                ok += 1
            else:
                bad.append((r["task"], r["output"], got))
            print(f"\r{i}/{len(rows)}", end="", flush=True)
        print()
        print(f"精确匹配 {ok}/{len(rows)} = {ok / len(rows):.0%}")
        for task, gold, got in bad[:5]:
            print("-" * 60)
            print("task:", task)
            print("gold:", gold)
            print("got :", got)
        return

    task = args.task or demo["task"]
    print("task   :", task)
    print("route  :", route(tok, model, demo["manifest"], task))
    print("gold   :", demo["output"])


if __name__ == "__main__":
    sys.exit(main())
