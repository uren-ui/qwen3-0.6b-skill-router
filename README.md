# Qwen3-0.6B Skill Router · 协调模型（LoRA）

> 把「用户的一句话 + 当前可用技能清单」直接路由成一条结构化决策：**调用哪个技能、带什么参数**，还是**反问一句**，还是**直接拒绝**。
> 基座 Qwen3-0.6B，LoRA 只训练 10,092,544 个参数（占全模型 1.665%），RTX 3080 单卡 53 分钟训完，产物 38.5 MB。

[![base](https://img.shields.io/badge/base-Qwen3--0.6B-blue)]() [![peft](https://img.shields.io/badge/peft-LoRA%20r16-green)]() [![license](https://img.shields.io/badge/license-Apache--2.0-lightgrey)]()

---

## 它解决什么问题

本地多智能体系统里，"协调层"要把一句人话翻译成机器能执行的动作。用 4B/7B 模型做这件事，显存被占掉一大半，真正干活（出图、出视频、跑代码）的模型就没地方放了。

这个 adapter 的思路是：**协调只需要判断，不需要生成能力**。所以把它压到一个 0.6B 的小模型上，让它专心做一件事——读技能清单，输出一条 JSON。判断对了，后面交给专门的模型；判断错了，后面再强也没用。

## 长什么样

### 输入（两条消息）

**system**：角色约束 + 输出格式（`call / ask / refuse` 三类，外加"只能做一半"带 `note` 的变体），以及几条硬规则：

- 任务里多个诉求时，先做能做的部分，不要因为附带的小要求就整条拒绝
- 技能必须和任务直接相关才能 `call`，不能为了有输出就硬塞技能
- 缺关键信息（没给路径、没说哪个模型）就 `ask`，不许猜

**user**：任务 + 可用技能清单（manifest）。清单里每个技能都带**成功率和中位耗时**，还带**踩过的坑**——这些是运行时从执行记录里长出来的，不是写死的文档：

```
可用技能：
- switch_model（成功率1.0，中位7.98s）：把本地模型服务切换成指定模型……
  参数：model（必填）：qwen3-0.6b 或 qwen2.5-0.5b；port（可选）：端口号，默认 8090
  边界：只做"切换"。不能只停止不启动，也不能用它查询当前状态
- write_svg（成功率0.5，中位6.39s，坑:环境不满足×1）：画一张图存到文件……
- model_status（还没有执行记录）：查当前哪个模型在线、占多少显存……
- generate_image（成功率1.0，中位3.04s）：画一张图片（PNG）存成文件……
```

**任务**：`给我画下雨天的街道，存到 D:\ai-lab\pelican.svg`

### 输出（一行 JSON）

```json
{"action":"call","skill":"write_svg","params":{"spec":"下雨天的街道","path":"D:\\ai-lab\\pelican.svg"}}
```

四种形态：

| 输出 | 含义 |
|---|---|
| `{"action":"call","skill":"…","params":{…}}` | 能完整做完 |
| `{"action":"call","skill":"…","params":{…},"note":"…"}` | 只能做一部分，note 写剩下的为什么做不了 |
| `{"action":"ask","need":"…"}` | 缺信息，问一句就能做，**不许猜** |
| `{"action":"refuse","reason":"…"}` | 技能清单里压根没有这个能力 |

关键设计：**拒绝和反问是正样本，不是失败**。宁可让模型说"我不知道"，也不要它编一个不存在的技能名字出来。训练数据里 `refuse` 占 20%、`ask` 占 7.7%，就是为了让它学会闭嘴。

## 快速开始

```bash
pip install torch transformers peft
```

```python
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3-0.6B"
ADAPTER = "./adapter"          # 本仓库的 adapter/ 目录

tok = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(model, ADAPTER).eval()

SYSTEM = (
    "你是电脑协调器。你只有下面列出的技能，不要发明技能。\n"
    "只输出一行 JSON，不要解释，不要复述任务。\n"
    "能做完：{\"action\":\"call\",\"skill\":\"技能名\",\"params\":{...}}\n"
    "只能做完一部分：{\"action\":\"call\",\"skill\":\"技能名\",\"params\":{...},\"note\":\"剩下的做不了的原因\"}\n"
    "缺信息、问一句就能做：{\"action\":\"ask\",\"need\":\"要问什么\"}\n"
    "压根没有这个能力：{\"action\":\"refuse\",\"reason\":\"短原因\"}\n"
)

user = "manifest 内容…\n\n任务：给我画下雨天的街道，存到 D:\\ai-lab\\pelican.svg"
msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
out = model.generate(**tok(text, return_tensors="pt").to(model.device),
                     max_new_tokens=120, do_sample=False, pad_token_id=tok.eos_token_id)
print(tok.decode(out[0][tok(text, return_tensors="pt").input_ids.shape[1]:], skip_special_tokens=True))
```

`enable_thinking=False` 是必须的——训练时就是关掉思考模式的，开了反而掉准确率。

完整可跑的例子见 [`examples/infer.py`](examples/infer.py)。

## 训练

| 项 | 值 |
|---|---|
| 基座 | `Qwen/Qwen3-0.6B`（Apache-2.0） |
| 方法 | LoRA，手写训练循环（**不用 trl / bitsandbytes**） |
| rank / alpha / dropout | 16 / 32 / 0.05 |
| target modules | `q k v o gate up down proj`（7 个） |
| 可训练参数 | 10,092,544 / 606,142,464 = **1.665%** |
| 精度 | bf16 全精度（0.6B 放得下，没必要 4bit） |
| 优化器 | AdamW，lr 2e-4，wd 0.01，cosine + 5% warmup，grad clip 1.0 |
| batch / max_len | 4 / 512 tokens |
| epochs | 3（5,082 步）→ 最终版 2,547 步 |
| 数据 | 3,394 条训练 / 312 条验证（取前 60 条评测） |
| 硬件 | RTX 3080 20GB，**53.3 分钟** |

完整超参、每一轮迭代的 loss 与验证集曲线、以及踩过的坑，见 [`training/TRAINING_LOG.md`](training/TRAINING_LOG.md)。

## 评测

指标是**输出 JSON 的精确匹配**（不是字段级，也不是语义相似）：**去掉所有空格后**逐字符比对 `call` 的技能名与参数、`ask` 的 need、`refuse` 的 reason。

| 验证集 | 结果 |
|---|---|
| dev 前 60 条（每 epoch 末评估） | epoch1 58/60 (97%) → epoch2 59/60 (98%) → epoch3 **60/60 (100%)** |

这个 100% 要打个折看：验证集和训练集同分布（同一套合成流程产出的），60 条也太少。更需要关注的是这几个数字背后的事实——**loss 到第 3 个 epoch 已经掉到 0.0012**，说明它在背格式，不是在学推理。技能清单换了、任务句式换了，就得重训。

更多说明见 [`eval/EVAL.md`](eval/EVAL.md)。

## 目录

```
adapter/                  LoRA 权重 + tokenizer（38.5MB 权重 + 11MB tokenizer）
training/
  train_judge.py          训练脚本（手写循环，Windows 上跑得通）
  run_train.ps1           一条命令复现
  data/
    train_mix.jsonl       3,394 条训练
    dev_mix.jsonl         312 条验证
    dev_eval_60.jsonl     评测用的那 60 条（dev_mix 前 60）
    README.md             数据格式说明 + 样例
  logs/                   12 份原始训练日志（含每次迭代）
  TRAINING_LOG.md         训练记录：迭代表、曲线、坑
eval/EVAL.md              评测口径与结果
examples/infer.py         最小可跑推理示例
```

## 适用边界（先看这段，别踩）

**能用的场景**
- 技能清单是**运行时注入的**（不是让模型记住技能列表）——换清单重训或直接换更大模型
- 任务短（训练数据平均 23.6 字，最长 56 字），一次一句话
- 要的是"判断 + 挑参数"，不是"写代码""解释原理"

**不能用的场景**
- 技能池大变（新增十几个技能、语义边界重叠）→ 零样本泛化基本没有，必须补数据重训
- 需要多轮对话、需要记住上文、需要解释自己为什么这么判断
- 需要它生成内容（写文案、写代码、画图）——它只会输出一条路由 JSON，其它一律不行

**已知偏差**
- 训练数据是程序合成的，句式模板化严重：真实用户说话一旦绕个弯，容易错判成 `ask` 或 `refuse`
- `combo:*` 这类组合技能样本只有几十条，几乎学不到，不建议依赖
- 中英混输、口语化、错别字都没覆盖

## 许可

- Adapter 权重与训练数据：**Apache-2.0**（随基座许可）
- 基座模型 `Qwen/Qwen3-0.6B` 遵循其原始许可
- 训练数据由本机运行记录合成，不含第三方数据、不含个人信息

## 引用

```
@misc{qwen3_06b_skill_router_2026,
  title  = {Qwen3-0.6B Skill Router: a LoRA adapter for local skill routing},
  author = {Local AI Lab},
  year   = {2026},
  note   = {LoRA adapter on Qwen/Qwen3-0.6B, trained on 3,394 synthetic routing samples}
}
```
