# Upload Checklist

The local project is intended to be uploaded in three parts.

## 1. GitHub Project

Upload the repository source, including:

- `src/`
- `scripts/`
- `configs/`
- `app.py`
- `requirements.txt`
- `pyproject.toml`
- `README.md`

Do not upload temporary vector downloads. They are ignored by `.gitignore`.

## 2. HuggingFace Dataset

Create the dataset package:

```powershell
$PY="C:\Users\marcu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $PY scripts/package_dataset.py
```

Upload the contents of `dist/dataset/` to the HuggingFace dataset repository.

## 3. HuggingFace Space

Create the Space package:

```powershell
$PY="C:\Users\marcu\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $PY scripts/package_space.py
```

Upload the contents of `dist/space/` to a Gradio HuggingFace Space.

When this step begins, provide:

- GitHub repository URL or desired owner/name
- HuggingFace username or organization
- HuggingFace dataset repository name
- HuggingFace Space repository name
- credentials or authenticated CLI/session access
