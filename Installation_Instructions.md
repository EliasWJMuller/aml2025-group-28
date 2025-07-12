# Instructions for MPS Devices
## RNAgrail
1. Create and activate a virtual environment (tested with Python 3.13.3)
2. Upgrade pip to the latest version

``` 
cd RNAgrail
```

### Installing build tools and dependencies for PyTorch Geometric
```
xcode-select --install
brew install cmake llvm libomp

export CC=$(brew --prefix llvm)/bin/clang
export CXX=$(brew --prefix llvm)/bin/clang++
export LDFLAGS="-L$(brew --prefix llvm)/lib"
export CPPFLAGS="-I$(brew --prefix llvm)/include"

export MACOSX_DEPLOYMENT_TARGET=$(sw_vers -productVersion)

brew install meson ninja  
pip install meson-python

pip install torch torchvision torchaudio   

pip install torch_geometric             

pip install --no-build-isolation git+https://github.com/pyg-team/pyg-lib.git
pip install --no-build-isolation torch-scatter
pip install --no-build-isolation torch-sparse
pip install --no-build-isolation torch-cluster
pip install --no-build-isolation torch-spline-conv
 # Additionally, upgrade scikit-learn if needed
pip install --upgrade scikit-learn
pip install lightning==2.2.0
```

In your pyproject.toml file, update the torch dependency to:
```
torch = "==2.6.0" -> torch = ">=2.3.0,<3.0.0"
```
and remove flash-attn from the dependencies list

```
pip install --no-build-isolation .
```

## Rinalmo
Before running RNAgrail, clone and configure RiNALMo in your virtual environment; this ensures compatibility when RNAgrail invokes RiNALMo.

```
git clone https://github.com/lbcb-sci/RiNALMo
cd RiNALMo
````

In the conda environment file, remove the following CUDA-specific dependencies:

```
  - cuda-cudart=11.8.89=0
  - cuda-cupti=11.8.87=0
  - cuda-libraries=11.8.0=0
  - cuda-nvrtc=11.8.89=0
  - cuda-nvtx=11.8.86=0
  - cuda-runtime=11.8.0=0
  - *=py3.11_cuda11.8_cudnn8.7.0_0
  - pytorch-cuda=11.8=h7e8668a_5
  - pytorch-mutex=1.0=cuda
```
Add the flash-attn library to the dependencies:
```
  - flash-attn
```

In the pip section of the environment file, remove the flash-attn entry:

```
- flash-attn==2.3.2
```
If the flash_attn package directory is missing, create a placeholder module:
```
mkdir -p $(python -c "import site; print(site.getsitepackages()[0])")/flash_attn
touch $(python -c "import site; print(site.getsitepackages()[0])")/flash_attn/__init__.py
```

Install the RiNALMo dependencies into the RNAgrail virtual environment:

```
pip install --no-build-isolation .
# or manually if it does not work
pip install \
  torch torchvision torchaudio \
  numpy==1.26.4 \
  'scikit-learn>=1.4.0' \
  pandas scipy \
  'biopython>=1.83' \
  rnapolis==0.3.11 \
  torch-sparse==0.6.18 torch-scatter==2.1.2 torch-cluster==1.6.3 \
  pytorch-lightning==2.2.0.post0 lightning-utilities==0.10.1 torchmetrics==1.2.1 \
  wandb==0.15.12
```

Download the pretrained RiNALMo model and run secondary structure prediction (test only):

```
brew install wget
wget https://zenodo.org/records/15043668/files/rinalmo_giga_pretrained.pt

python train_sec_struct_prediction.py \
  ./ss_data --test_only \
  --init_params ./weights/rinalmo_giga_ss_archiveII-5s_ft.pt \
  --dataset archiveII_5s --prepare_data \
  --output_dir ./outputs/archiveII/5s/ \
  --accelerator mps --devices 1
```

## Back to RNAgrail

```
wget "https://zenodo.org/records/13757098/files/model_epoch_800.tar.gz?download=1" \
     -O model_epoch_800.tar.gz
```

Test RNAgrail inference with a sample .dotseq file:

```
grapharna --input={some .dotseq file}
```

When testing, use the modified source files directly rather than the CLI.
First, download and unpack the test and validation datasets from Zenodo into the appropriate directories.
Alternatively, prepare the datasets using the provided preprocessing script:

```
python preprocess_rna_pdb.py --pdb_dir=<pdb_directory> --save_dir=data/<dataset name> --save_name=<train-pkl / test-pkl / val-pkl>

```

```
time python -m grapharna.main_rna_pdb_single \
  --epochs 1 \
  --batch_size 1
```











