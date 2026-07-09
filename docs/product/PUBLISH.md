# Publication Commands

## 1. GitHub (public repo)

```bash
# Create public repo (Pepe confirms name/org)
gh repo create FrameLanguageLM --public --source=. --push

# Or if repo already exists:
git remote add origin https://github.com/PEPE_USERNAME/FrameLanguageLM.git
git push -u origin master
```

## 2. HuggingFace (model upload)

```bash
# Login with HF token
huggingface-cli login

# Upload model repo (~348 MB)
huggingface-cli upload PEPE_USERNAME/FrameLanguageLM hf_repo/ . --repo-type model
```

## 3. PyPI (CLI package) — optional, later

```bash
uv build
uv publish
```

## Notes
- Replace PEPE_USERNAME with actual GitHub/HF username
- GitHub repo creation requires `gh` CLI authenticated
- HF upload requires a write token from huggingface.co/settings/tokens
- PyPI publish requires a PyPI API token
