# GitHub Transfer Commands

This project is prepared for GitHub transfer through a private repository. The current local Codex session can edit files, but Git metadata writes are blocked by Windows permission errors such as:

- `Unable to create .git/index.lock: Permission denied`
- `Unable to create .git/refs/heads/robust-imperfect-csi.lock: Permission denied`

Run the following commands from a terminal that has normal write access to the repository.

## 1. Create Private GitHub Repository

Create an empty private GitHub repository named:

```text
imperfect-csi-swinjscc
```

Do not initialize it with README, license, or `.gitignore`.

## 2. Prepare Local Branch

From:

```powershell
C:\Users\Admin\Desktop\swimjscc_f\不完美信道\从vscode传来的文件\SwinJSCC\SwinJSCC
```

run:

```powershell
git switch -c robust-imperfect-csi
```

If the branch already exists:

```powershell
git switch robust-imperfect-csi
```

## 3. Add Transfer Remote

Replace `<repo-url>` with your new private repository URL:

```powershell
git remote add work <repo-url>
```

If `work` already exists:

```powershell
git remote set-url work <repo-url>
```

## 4. Stage Only Safe Files

The `.gitignore` added in this workspace excludes checkpoint, dataset, history, visualization outputs, caches, logs, and LaTeX build products.

Stage only the intended transfer scope:

```powershell
git add .gitignore main.py net/network.py utils.py eval_msssim_cpu.py AGENT.md chapters figures latex mismatch_results plan tables
```

Check staged content:

```powershell
git status --short
git diff --cached --stat
```

Confirm that these directories are not staged:

```text
checkpoint/
dataset/
history/
__pycache__/
.ipynb_checkpoints/
mismatch_vis_cifar10/
```

## 5. Commit and Push

```powershell
git commit -m "Add imperfect-CSI SwinJSCC draft and robust training groundwork"
git push -u work robust-imperfect-csi
```

## 6. Pull on Remote Server

On the remote server:

```bash
cd /root/autodl-tmp/SwinJSCC/SwinJSCC
git remote add work <repo-url>
git fetch work
git switch -c robust-imperfect-csi work/robust-imperfect-csi
```

If the remote already has uncommitted changes, inspect first:

```bash
git status --short
```

Do not overwrite remote checkpoint or dataset directories. They should remain local to the training machine.
