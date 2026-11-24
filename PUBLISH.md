# 发布到 GitHub 指南

这个目录已经是一个**可直接推送的仓库内容**（README、LICENSE、.gitignore、.gitattributes 都在位），只差 `git init` 和 push。

## 关键限制（先看，避免白折腾）

| 限制 | 说明 |
|---|---|
| GitHub 网页上传单文件 ≤ **25 MB** | `adapter/adapter_model.safetensors` 是 **38.5 MB**，网页拖拽会直接报错 |
| GitHub 单文件硬上限 **100 MB** | 38.5 MB 没超，所以**不需要 LFS**，普通 git 推得动 |
| 整个仓库约 **52 MB** | 其中权重 38.5 MB + tokenizer 11 MB |

所以：**权重必须用 git / GitHub Desktop 推，不能网页上传。**

---

## 方式 A：GitHub Desktop（最省事，不需要命令行）

1. 下载安装 [GitHub Desktop](https://desktop.github.com/)，登录你的账号
2. 菜单 `File → Add local repository`，选这个目录 `D:\ai-lab\_release\qwen3-0.6b-skill-router`
3. 它会提示 "This directory does not appear to be a Git repository. Create a repository?" → 点 **create a repository**
4. 填仓库名（建议 `qwen3-0.6b-skill-router`），点 **Create Repository**
5. 左下角填 commit 信息（例如 `Initial commit: Qwen3-0.6B skill router LoRA`），点 **Commit to master**
6. 右上角 **Publish repository**，取消勾选 "Keep this code private"（如果你要公开）

## 方式 B：命令行（需先装 git）

```powershell
winget install --id Git.Git -e --source winget
# 装完重开一个终端
```

```powershell
cd D:\ai-lab\_release\qwen3-0.6b-skill-router
git init -b main
git add .
git commit -m "Initial commit: Qwen3-0.6B skill router LoRA"
```

在 GitHub 上先**手动新建一个空仓库**（不要勾 Add README / .gitignore / license，避免冲突），然后：

```powershell
git remote add origin https://github.com/<你的用户名>/qwen3-0.6b-skill-router.git
git push -u origin main
```

首次 push 会弹浏览器登录 GitHub，授权即可。

> 如果提示 `remote: Support for password authentication was removed`，说明你用密码认证了。改用：
> `gh auth login`（需装 GitHub CLI），或把 remote 换成 SSH：
> `git remote set-url origin git@github.com:<你的用户名>/qwen3-0.6b-skill-router.git`

## 方式 C：只有网页，也想放权重

两条路：

1. **把权重发到 Release**：网页上传 zip 到 Release 附件（Release 附件上限 2 GB，不受 25 MB 限制），仓库里只放代码和文档，README 里写一句"权重见 Release"
2. **改用 Hugging Face**：HF 对模型仓库更合适，网页可以直接传大文件，还自带模型页渲染 `MODEL_CARD.md` 的 YAML 头部。要发 HF 的话，把 `MODEL_CARD.md` 改名成 `README.md` 放在上层即可（这样 GitHub 就没 README 了，两份仓库要分开）

---

## 推之前该确认的三件事

1. **`adapter/adapter_config.json` 里的 `base_model_name_or_path`** 已经改成 `Qwen/Qwen3-0.6B`（原先是本机绝对路径 `D:\ai-lab\models\...`）。这是发布必须改的，否则别人加载会找不到路径。
2. **`training/train_judge.py` 的 `MODEL_PATH`** 也已改成 `os.environ.get('BASE_MODEL') or 'Qwen/Qwen3-0.6B'`。
3. **数据里含本机路径**（`D:\ai-lab\...`）。它只是任务文本的一部分，不含个人信息、不含第三方数据。如果你介意暴露目录结构，可以用脚本批量替换成 `D:\work\...` 之类再推——但注意：**替换后要与训练数据保持同一份**，否则复现结果对不上。

## 建议顺手加的仓库设置

- **Topics**（GitHub 仓库页右上角）：`lora` `peft` `qwen3` `llm-router` `local-ai` `agents`
- **About** 里填一句：`A 0.6B LoRA that routes natural-language tasks to a runtime-injected skill list.`
- **LICENSE** 已放 Apache-2.0 全文（继承基座 Qwen3-0.6B 的许可）
- 如果想要 badge，README 顶部已有 base/peft/license 三个静态 badge

## 发布后自查

- 仓库首页能正常渲染 README（检查表格、代码块、锚点链接）
- 点开 `adapter/` 能看到 `adapter_model.safetensors`（38.5 MB，GitHub 会显示 "Stored with Git LFS" 与否都正常）
- `examples/infer.py` 里的相对路径指向 `../adapter` 与 `../training/data`，克隆下来直接跑得通
