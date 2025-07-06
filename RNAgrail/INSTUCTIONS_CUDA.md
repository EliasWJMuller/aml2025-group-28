# Instructions to Set Up RNAGrail Training on a CUDA Machine

This guide provides a comprehensive set of instructions for installing RNAGrail and preparing datasets for training on a CUDA-enabled machine.

**Note:** This library and set of instructions assume you have a CUDA machine with an architecture compatible with flash-attn==2.3.2 (e.g., an NVIDIA A100 GPU).


## Installation
This section outlines the steps to install all necessary dependencies and libraries for RNAGrail.

### Installing Core Dependencies

Begin by installing the fundamental Python packages required for the project, including PyTorch with CUDA support.
```bash
pip install torch==2.3.0+cu121
pip install "setuptools<65.0.0"
pip install torchvision
pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```

### Installing Flash-Attention

Next, install the `flash-attn` library, which is crucial for optimized attention mechanisms in deep learning models. Ensure the `USE_FLASH_ATTENTION` environment variable is set and CUDA architecture is specified.

```bash
set USE_FLASH_ATTENTION=1
pip install build
pip install cmake
pip install ninja
pip install wheel
export MAX_JOBS=$(($(nproc)/2))
export TORCH_CUDA_ARCH_LIST="8.0"

pip install flash_attn flash-attn==2.3.2
```

### Installing RNAGrail

Now, clone the GraphaRNA repository and install RNAGrail in editable mode.

```bash
cd ~
pip install setuptools<65.00
git clone git@github.com:mjustynaPhD/GraphaRNA.git
cd GraphaRNA
pip install -e .

```

### Installing RiNALMo

Proceed to clone and install the RiNALMo library.

```bash
git clone https://github.com/lbcb-sci/RiNALMo
cd RiNALMo
pip install -e .
```

### Miscellaneous Dependencies

Finally, install additional utility libraries required for data handling and scientific computing.

```bash
pip install --upgrade scikit-learn scipy requests
pip install pandas
```


## Download and Pre-Process Kaggle Data
This section details how to download and prepare the Kaggle Stanford RNA 3D Folding dataset for use with RNAGrail.

First, install the Kaggle API and set up your authentication.

```bash
cd ~
pip install kaggle
mkdir ~/.kaggle
touch ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Action Required: Retrieve an API key from Kaggle and paste it into the kaggle.json file you just created.**

Download and unzip the Kaggle competition data.

```bash
cd ~
kaggle competitions download -c stanford-rna-3d-folding
mkdir stanford-rna-3d-folding
mv stanford-rna-3d-folding.zip stanford-rna-3d-folding
cd stanford-rna-3d-folding
unzip stanford-rna-3d-folding.zip
```

Prepare the downloaded dataset into a format suitable for RNAGrail's preprocessing pipeline.

```bash
python ~/aml2025-group-28/data_prep/dataset_prep.py
```


Now, preprocess the Kaggle dataset for consumption by GraphaRNA, specifying the input directories, output names, file types, and the number of CPUs to use for parallel processing.

```bash
cd ~/aml2025-group-28/RNAgrail/src/grapharna
python preprocess_rna_pdb.py --pdb_dir=/home/ubuntu/stanford-rna-3d-folding/train_cif/ --save_dir=data/all_cifs --save_name=train-pkl --file_type=.cif --num_cpus=100
python preprocess_rna_pdb.py --pdb_dir=/home/ubuntu/stanford-rna-3d-folding/test_cif/ --save_dir=data/all_cifs --save_name=test-pkl --file_type=.cif --num_cpus=100
python preprocess_rna_pdb.py --pdb_dir=/home/ubuntu/stanford-rna-3d-folding/val_cif/ --save_dir=data/all_cifs --save_name=val-pkl --file_type=.cif --num_cpus=100
```



## Download and Preprocess Original Data from Paper
This section guides you through downloading and preparing the original dataset used in the paper, which is hosted on Zenodo.

Navigate to your home directory, create a new directory for the original data, and then download and extract the compressed archives containing the rRNA/tRNA and non-rRNA/tRNA datasets.
```bash
cd ~
mkdir og_data
cd og_data
curl "https://zenodo.org/records/13750967/files/rRNA_tRNA.tar.gz?download=1" -o rRNA_tRNA.tar.gz
tar -xvzf rRNA_tRNA.tar.gz
curl "https://zenodo.org/records/13750967/files/non_rRNA_tRNA.tar.gz?download=1" -o non_rRNA_tRNA.tar.gz
tar -xvzf non_rRNA_tRNA.tar.gz
```

Preprocess the original dataset for consumption by GraphaRNA, similar to the Kaggle data. Note the assumption regarding the validation split.

```bash
python preprocess_rna_pdb.py --pdb_dir=/home/ubuntu/og_data/rRNA_tRNA/ --save_dir=data/og_new --save_name=train-pkl --file_type=.pdb --num_cpus=100
python preprocess_rna_pdb.py --pdb_dir=/home/ubuntu/og_data/non_rRNA_tRNA/ --save_dir=data/og_new --save_name=test-pkl --file_type=.pdb --num_cpus=100
# note that the paper explains which dataset is the train and which the test split, but does not mention val_split. I am therefore assuming test==val
python preprocess_rna_pdb.py --pdb_dir=/home/ubuntu/og_data/non_rRNA_tRNA/ --save_dir=data/og_new --save_name=val-pkl --file_type=.pdb --num_cpus=100
```



## Download Model, run Inference and Training

### Download the Model Weights
This section describes how to download a pre-trained model checkpoint from Zenodo.

Navigate to the RNAGrail directory, create a save directory, and then download and extract the pre-trained model.

```bash
cd ~/aml2026-group-28/RNAgrail
mkdir -p save/grapharna
wget https://zenodo.org/records/13750967/files/model_epoch_800.tar.gz?download=1 -O model_epoch_800.tar.gz
tar -xvzf model_epoch_800.tar.gz && mv model_800.h5 save/grapharna/
```

### Try an Inference Run
Perform a quick inference run using the grapharna command with a sample input file to verify the setup.

```bash
grapharna --input=/home/ubuntu/aml2025-group-28/inputoutputtesting/dotseqs/1A1T_B.dotseq
```

### Run a Single Training Epoch with Original Data
Execute a single training epoch using the original dataset. This command utilizes torchrun for distributed training.

```bash
cd /home/ubuntu/aml2025-group-28/RNAgrail/src/grapharna/
torchrun --nproc_per_node=8 main_rna_pdb.py --dataset=og_new --epochs 1 --batch_size 8 --dim=64 --n_layer=2 --lr=5e-4 --timesteps=500 --mode=coarse-grain --knn=2 --lr-step=30
```

This command should execute normally and produce output similar to the following, indicating the training progress:

```bash
Epoch: 0, step: 0, loss: 1.321
...
Epoch: 0, step: 19, loss: 0.0667
Epoch: 1, Loss: nan, Denoise Loss: nan, Val Loss: nan, Val Denoise Loss: nan, LR: 0.0005
```

### Run a Single Training Epoch with Kaggle Data
Attempt to run a single training epoch using the Kaggle dataset. This is expected to encounter an error related to memory limitations.

```bash
cd /home/ubuntu/aml2025-group-28/RNAgrail/src/grapharna/
torchrun --nproc_per_node=8 main_rna_pdb.py --dataset=all_cifs --epochs 1 --batch_size 8 --dim=64 --n_layer=2 --lr=5e-4 --timesteps=500 --mode=coarse-grain --knn=2 --lr-step=30
```

This command is expected to result in an error similar to:
```bash
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
main_rna_pdb.py FAILED
```

This error is likely related to out-of-memory issues, as the original architecture scales to meet the dataset size, potentially requiring around 500GB of GPU memory. Addressing this would require significant engineering effort.


