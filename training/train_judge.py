"""LoRA 微调 Qwen3-0.6B 做判断层。

不用 trl / bitsandbytes：
- 0.6B 太小，不需要 4bit；fp16/bf16 全精度放得下
- 手写训练循环，避开 Windows 上 bnb 的坑，也便于改

产物：lora_judge/（adapter，约 10~20MB）
"""

import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get('BASE_MODEL') or 'Qwen/Qwen3-0.6B'
OUT_DIR = os.environ.get('TRAIN_OUT') or os.path.join(HERE, 'lora_judge')
TRAIN_FILE = os.environ.get('TRAIN_FILE') or os.path.join(HERE, 'data', 'train.jsonl')
DEV_FILE = os.environ.get('DEV_FILE') or os.path.join(HERE, 'data', 'dev.jsonl')

# 批量模式：一次判多条，输出数组。
# 单独训一个 adapter，数据 100% 是数组格式——两种格式混着训最容易退化成固定答案。
BATCH_MODE = os.environ.get('JUDGE_BATCH_MODE') == '1'

MAX_LEN = int(os.environ.get('MAX_LEN') or (768 if BATCH_MODE else 512))
EPOCHS = 3
LR = 2e-4
BATCH = int(os.environ.get('BATCH') or 8)
WARMUP_RATIO = 0.05

SYSTEM_BATCH = (
    '你是电脑协调器。你只有下面列出的技能，不要发明技能。\n'
    '只输出一个 JSON 数组，不要解释，不要复述任务。\n'
    '这次会一次给你多条任务，每条前面带编号，你要按编号顺序一条一条判断。\n'
    '数组里每个元素是完整的 {...} 对象，一条任务一个对象，条数必须和任务数一样。\n'
    '每个对象只能是这四种之一：\n'
    '能做完：{"action":"call","skill":"技能名","params":{...}}\n'
    '只能做完一部分：{"action":"call","skill":"技能名","params":{...},"note":"剩下的做不了的原因"}\n'
    '缺信息、问一句就能做：{"action":"ask","need":"要问什么"}\n'
    '压根没有这个能力：{"action":"refuse","reason":"短原因"}\n'
    '判断标准：任务里有多个诉求时，先做能用技能做的那部分，'
    '不要因为附带的小要求就整条拒绝。\n'
    '任务必须和技能描述的事情直接相关才能 call，绝对不能为了有输出就硬塞技能。\n'
    '缺关键信息（没给路径、没说哪个模型）就 ask，不要猜。\n'
)

SYSTEM = SYSTEM_BATCH if BATCH_MODE else (
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


def user_text(sample):
    if BATCH_MODE:
        lines = [sample['manifest'], '任务列表：']
        lines += [f'[{i}] {t}' for i, t in enumerate(sample['tasks'], 1)]
        lines.append(f'按编号顺序输出 JSON 数组（长度={len(sample["tasks"])}）：')
        return '\n'.join(lines)
    return f'{sample["manifest"]}\n任务：{sample["task"]}\n输出 JSON：'


def load_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]


class JudgeDS(Dataset):
    def __init__(self, rows, tok):
        self.items = []
        self.skipped = 0
        for i, r in enumerate(rows):
            if i % 200 == 0:
                print(f'    tokenize {i}/{len(rows)}', flush=True)
            try:
                msgs = [
                    {'role': 'system', 'content': SYSTEM},
                    {'role': 'user', 'content': user_text(r)},
                ]
                prompt = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
                msgs = msgs + [{'role': 'assistant', 'content': r['output']}]
                full = tok.apply_chat_template(
                    msgs, tokenize=False, enable_thinking=False)

                p_ids = tok(prompt, add_special_tokens=False).input_ids
                f_ids = tok(full, add_special_tokens=False).input_ids
            except Exception as e:
                print(f'    第 {i} 条 tokenize 失败：{e}', flush=True)
                self.skipped += 1
                continue

            if len(f_ids) > MAX_LEN:
                self.skipped += 1
                continue
            if f_ids[:len(p_ids)] != p_ids:
                # 模板前缀不一致就跳过，不要悄悄错位
                self.skipped += 1
                continue

            labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
            self.items.append((f_ids, labels))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch, pad_id):
    maxlen = max(len(x[0]) for x in batch)
    ids, labels, mask = [], [], []
    for x, y in batch:
        pad = maxlen - len(x)
        ids.append(x + [pad_id] * pad)
        labels.append(y + [-100] * pad)
        mask.append([1] * len(x) + [0] * pad)
    return (torch.tensor(ids), torch.tensor(labels), torch.tensor(mask))


def evaluate(model, tok, rows, limit=60):
    model.eval()
    ok = 0
    total = 0
    for r in rows[:limit]:
        msgs = [
            {'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': user_text(r)},
        ]
        text = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        ids = tok(text, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=120, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        got = tok.decode(out[0][ids['input_ids'].shape[1]:],
                         skip_special_tokens=True).strip()
        total += 1
        if got.replace(' ', '') == r['output'].replace(' ', ''):
            ok += 1
    model.train()
    return ok, total


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print('[1] 加载模型（bf16）')
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, device_map='cuda')
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        bias='none',
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    print('[2] 构造数据集')
    train_rows = load_jsonl(TRAIN_FILE)
    dev_rows = load_jsonl(DEV_FILE)
    ds = JudgeDS(train_rows, tok)
    print(f'    训练样本 {len(ds)}（跳过 {ds.skipped}）  验证 {len(dev_rows)}')
    if len(ds) == 0:
        print('没有可用样本，退出')
        return

    dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    collate_fn=lambda b: collate(b, tok.pad_token_id))

    steps = len(dl) * EPOCHS
    warmup = max(1, int(steps * WARMUP_RATIO))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / warmup) * max(0.0, (steps - s) / max(1, steps - warmup)))

    print(f'[3] 开始训练  {len(dl)} 步/epoch × {EPOCHS} = {steps} 步')

    model.train()
    global_step = 0
    for ep in range(EPOCHS):
        ep_loss = 0.0
        for i, (ids, labels, mask) in enumerate(dl):
            ids = ids.to('cuda')
            labels = labels.to('cuda')
            mask = mask.to('cuda')
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            global_step += 1
            ep_loss += loss.item()
            if global_step % 20 == 0:
                print(f'    step {global_step}/{steps}  loss {loss.item():.4f}  '
                      f'lr {sched.get_last_lr()[0]:.2e}')
        print(f'  epoch {ep+1} 平均 loss {ep_loss/len(dl):.4f}')

        ok, total = evaluate(model, tok, dev_rows, limit=60)
        print(f'  >> 验证集精确匹配 {ok}/{total} = {ok/total:.0%}')

    print('[4] 保存 adapter')
    model.save_pretrained(OUT_DIR)
    tok.save_pretrained(OUT_DIR)
    size = sum(os.path.getsize(os.path.join(OUT_DIR, f))
               for f in os.listdir(OUT_DIR))
    print(f'    已保存到 {OUT_DIR}（{size/1024/1024:.1f}MB）')
    print(f'总耗时 {(time.time()-t0)/60:.1f} 分钟')


if __name__ == '__main__':
    main()
