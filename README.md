# FS25_AML_Group_28
Advanced Machine Learning Course FS25 @ UZH; Group 28; RNA-3D

# Installation (for macOS ARM)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/#upgrading-uv), either using curl or brew

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# or
brew install uv
```

Create a virtual env and install dependencies
```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch 
uv sync --no-build-isolation --no-cache # install dependencies from pyproject.toml file, resolving conflics
```

Note that the list of dependencies does not include the `flash-attn`, since that requires a cuda installation.

# Inference

Load the model weights
```bash
tar -xvzf model_epoch_800.tar.gz && mkdir -p RNAgrail/save/grapharna && mv model_800.h RNAgrail/save/grapharna
```
