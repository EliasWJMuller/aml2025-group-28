import os, gzip, shutil
from preprocess_rna_pdb import construct_graphs
import random

print(os.getcwd())

BASE          = "data/kaggle"
RAW_DIR       = os.path.join(BASE, "raw")
EXTRACTED_DIR = os.path.join(BASE, "extracted")
SAVE_DIR      = os.path.join(BASE, "out-pkl")

PCR = 0.1

archives = [
    os.path.join(RAW_DIR, f)
    for f in os.listdir(RAW_DIR)
    if f.endswith(".cif.gz")
]

print(f"Found {len(archives)} .cif.gz files in {RAW_DIR}")

# sort by size ascending
archives.sort(key=os.path.getsize)

# take smallest PCR fraction (at least one)
n_pcr = max(1, int(len(archives) * PCR))
to_extract = archives[:n_pcr]

print(f"Extracting {n_pcr} smallest files (PCR={PCR}) from {len(archives)} total archives")

# Clear out any old extracted files so we don't get name collisions
if os.path.isdir(EXTRACTED_DIR):
    shutil.rmtree(EXTRACTED_DIR)
os.makedirs(EXTRACTED_DIR, exist_ok=True)

# Now decompress into a clean folder
for gz in to_extract:
    base = os.path.basename(gz)[:-3]  # “1QZA.cif”
    outpath = os.path.join(EXTRACTED_DIR, base)
    with gzip.open(gz, "rb") as fi, open(outpath, "wb") as fo:
        shutil.copyfileobj(fi, fo)

# 5) build RNAGrail graphs
construct_graphs(
    seq_dir=None,
    pdbs_dir=EXTRACTED_DIR,
    save_dir=SAVE_DIR,
    save_name="",            # drop .pkl files straight into SAVE_DIR
    file_3d_type=".cif",
    extended_dotbracket=False,
    sampling=False
)

print("Done! .pkl graphs are in:", SAVE_DIR)

#assert that all pkl files are non-empty and contain valid data
for f in os.listdir(SAVE_DIR):
    with open(os.path.join(SAVE_DIR, f), 'rb') as file:
        data = file.read()
        if not data:
            raise ValueError(f"Validation Error: File {f} is empty!")
print(f"Validation Successful")

# Split out-pkl into val-pkl (20%) and train-pkl (80%)
VAL_DIR = os.path.join(BASE, "val-pkl")
TRAIN_DIR = os.path.join(BASE, "train-pkl")

# Clean and create output dirs
for d in [VAL_DIR, TRAIN_DIR]:
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

all_pkl = [f for f in os.listdir(SAVE_DIR) if f.endswith('.pkl')]
random.shuffle(all_pkl)
n_val = max(1, int(len(all_pkl) * 0.2))
val_files = all_pkl[:n_val]
train_files = all_pkl[n_val:]

for f in val_files:
    shutil.move(os.path.join(SAVE_DIR, f), os.path.join(VAL_DIR, f))
for f in train_files:
    shutil.move(os.path.join(SAVE_DIR, f), os.path.join(TRAIN_DIR, f))


print(f"Split {len(all_pkl)} .pkl files: {len(train_files)} in train-pkl, {len(val_files)} in val-pkl.")