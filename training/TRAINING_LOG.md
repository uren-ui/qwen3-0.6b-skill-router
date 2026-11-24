# 训练记录：Qwen3-0.6B Skill Router

这份记录不是"结果汇报"，而是**过程留痕**：一共训了 14 个 adapter、12 份日志，前 10 个基本都废了，留下的是第 14 个（`lora_roles4`）。失败的原因写在下面，比成功那轮更有用。

- 硬件：单卡 RTX 3080 20GB
- 环境：Windows 11 + Python 3.11 + PyTorch 2.x + PEFT 0.21.2
- 时间跨度：2026-11-12 ~ 2026-11-14（累计约 6 小时，其中 4.5 小时是这条线的训练）

---

## 一、两代模型

### 第一代：judge（单条判定，无技能清单）

| 产物 | 时间 | 说明 |
|---|---|---|
| `lora_judge_v1_backup` ~ `lora_judge_v3_backup` | 11-12 07:55 ~ 08:24 | 最初版本，各种格式试错 |
| `lora_r1` `lora_r2` `lora_r3` | 11-12 08:46 ~ 09:07 | 训练数据 257KB / 513KB / 641KB 逐版加大 |
| `lora_judge_v4` `lora_judge_v5` | 11-12 09:34 ~ 09:55 | 调超参 |
| `lora_judge` | 11-12 10:04 | 单条模式的定型版 |
| `lora_judge_batch` | 11-12 13:21 | 批量模式（一次判多条，输出 JSON 数组），独立训 |

第一代的根本问题：**技能是写死在 system prompt 里的**。技能一改就得重训，而且模型记不住"这个技能上次成功率只有 0.5"这种先验。所以有了第二代。

### 第二代：roles（技能清单运行时注入）

把技能清单从 system prompt 里挪出来，改成每条样本自带的 `manifest` 字段。清单里带成功率、中位耗时、踩过的坑——这些运行时从执行记录统计出来。模型变成"读清单做判断"，而不是"背技能表"。

| 日志 | 输出目录 | 训练样本 | 跳过 | 步/epoch | 总步数 | epoch 平均 loss | 每 epoch 验证 | 耗时 |
|---|---|---|---|---|---|---|---|---|
| `train_roles.log` | — | 2070 | 572 | — | 777 | — | — | 未完成 |
| `train_roles2.log` | `lora_roles` | 2642 | 0 | 331 | 993 | 0.0677 / 0.0031 / 0.0005 | 56→60→60 (93%→100%→100%) | 26.8 min |
| `train_roles3.log` | — | — | — | — | — | — | — | 中断 |
| `train_roles4.log` | — | 2022 | 648 | — | 759 | 0.0813 / — / — | — | 中断 |
| `train_roles5.log` | — | 2670 | 0 | 501 | 1002 | 0.0702 / 0.0029 | 58→60 (97%→100%) | — |
| `train_roles6.log` | `lora_roles` | 2650 | 20 | 663 | 1989 | 0.0649 / 0.0032 / 0.0005 | 59→60→60 (98%→100%→100%) | 28.2 min |
| `train_roles7.log` | `lora_roles` | 2670 | 0 | 668 | 2004 | 0.0647 / 0.0039 / 0.0005 | 59→60→60 (98%→100%→100%) | 30.9 min |
| `train_roles8.log` | `lora_roles2` | 3006 | 0 | 752 | 2256 | 0.0543 / 0.0030 / 0.0006 | 56→60→60 (93%→100%→100%) | 45.4 min |
| `train_roles9.log` | — | — | — | — | — | — | — | 中断 |
| `train_roles10.log` | `lora_roles3` | 3058 | 0 | 765 | 2295 | 0.0571 / 0.0046 / 0.0012 | 57→60→60 (95%→100%→100%) | 41.6 min |
| `train_roles12.log` | **`lora_roles4`（发布版）** | 3394 | 0 | 849 | 2547 | **0.0526 / 0.0030 / 0.0012** | **58→59→60 (97%→98%→100%)** | **53.3 min** |

> `train_roles11.log` 与 12 是同一配置，11 那次中断了（步数同样 2547），12 跑完。

样本数从 2642 一路加到 3394，是所有失败版本里唯一持续有效的改进手段——**加数据比调超参有用**。

---

## 二、最终版超参（`lora_roles4`）

```text
base              Qwen/Qwen3-0.6B
quantization      none（bf16 全精度）
lora_r            16
lora_alpha        32
lora_dropout      0.05
target_modules    q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
bias              none
trainable         10,092,544 / 606,142,464 = 1.6650%
dtype             bfloat16
optimizer         AdamW(lr=2e-4, weight_decay=0.01)
scheduler         LambdaLR: 5% 线性 warmup + 线性衰减到 0
grad_clip         1.0
batch_size        4
max_len           512（只保留整段 ≤512 token 的样本，超了直接丢）
epochs            3
steps             2547（849 × 3）
train samples     3394
dev samples       312（取前 60 条评测）
decode            greedy（do_sample=False），max_new_tokens=120
eval metric       输出 JSON 精确匹配（逐字符）
```

**为什么手写训练循环、不用 trl**：0.6B 不需要 4bit，也就不需要 bitsandbytes；Windows 上 bnb 的坑一堆。手写 244 行反而看得清每一步在干什么，出问题好定位。

---

## 三、踩过的坑（按严重程度排）

### 1. 数据格式混训 → 模型退化成一个固定答案

最初把"单条 JSON"和"批量 JSON 数组"两种格式混在一份数据里训。结果模型学到一个捷径：不管问什么都输出同一个数组骨架。`lora_judge_batch` 就是为此拆出去的——**一种输出格式，一个 adapter**。宁可多训一个模型，也不要混。

### 2. `MAX_LEN` 截断样本 → 训练/验证全乱

`train_roles.log` 跳过 572 条、`train_roles4.log` 跳过 648 条（占总样本 20%+）。原因：manifest 平均 1274 字，加上 system prompt 和任务，很容易超 512 token。当时的处理是"超了就丢"——**丢掉的都是长清单样本**，等于让模型只见过短清单，一到真实环境（十几个技能）就崩。

后来改成：① 精简 manifest 措辞；② 按长度分层抽样，保证长清单在训练集里有代表；③ 用 `skipped` 计数当体检指标，每轮必须看。

### 3. chat template 前缀错位，静默训坏

手拼 `prompt` 和 `full` 两段文本再 tokenize，如果两者前缀不一致（多一个空格、少一个 special token），`labels` 就会整体错位一格——loss 照样下降，但学的是错的东西。

修法：tokenize 完直接断言 `f_ids[:len(p_ids)] == p_ids`，不相等就跳过并计数。这个断言一直留在脚本里。

### 4. 不关 thinking 模式，准确率掉

Qwen3 默认带思考链。训练时用 `enable_thinking=False`，推理时也必须关。开着训/关着推（或反过来）会造成分布不一致，实测准确率明显下降。

### 5. loss 掉到 0 → 不是好消息

第 3 个 epoch loss 0.0012，验证 100%。看数字很漂亮，但这是**背书**不是**学推理**。证据：换一批句式稍微绕一点的任务，立刻错判成 `ask`。所以这版模型的定位只能是"格式稳定的路由器"，不能当判断器宣传。

### 6. `alpha=32 / r=16`（缩放系数 2）的副作用

`lora_alpha=32, r=16` 相当于放大 2 倍，前期收敛快，但也更容易过拟合到模板。试过 `alpha=16`，前两个 epoch 验证掉到 93%，最终还是回到 32 + 早停（第 3 epoch 保存）。

### 7. 验证集只有 60 条，且与训练同分布

`evaluate()` 固定取 dev 前 60 条，每 epoch 算一次精确匹配。60 条里错 1 条就是 1.7 个百分点，波动很大，**不能用它做模型选择**。它唯一的作用是"确认训练没崩"。

---

## 四、复现

```powershell
# 依赖
pip install torch transformers peft

# 训练（默认读 training/data/train_mix.jsonl 与 dev_mix.jsonl）
$env:BASE_MODEL = "Qwen/Qwen3-0.6B"     # 或本地路径
$env:TRAIN_FILE = "training/data/train_mix.jsonl"
$env:DEV_FILE   = "training/data/dev_mix.jsonl"
$env:TRAIN_OUT  = "out/lora_router"
$env:BATCH      = "4"
python training/train_judge.py
```

或者直接跑 `training/run_train.ps1`（已按上面的默认值写好）。

> 脚本里的 `OUT_DIR` 默认写到 `./lora_judge`，发布版的权重来自 `TRAIN_OUT=lora_roles4` 那次。超参完全一致，所以复现出的 adapter 应该能到 97% 以上。

## 五、文件说明

| 文件 | 内容 |
|---|---|
| `training/train_judge.py` | 训练脚本（244 行，含数据构造、手写训练循环、逐 epoch 精确匹配评测） |
| `training/run_train.ps1` | 复现命令 |
| `training/data/train_mix.jsonl` | 3,394 条训练样本 |
| `training/data/dev_mix.jsonl` | 312 条验证样本 |
| `training/data/dev_eval_60.jsonl` | 评测实际用的前 60 条 |
| `training/logs/train_roles*.log` | 12 份原始日志，`train_roles12.log` 是发布版那次 |
