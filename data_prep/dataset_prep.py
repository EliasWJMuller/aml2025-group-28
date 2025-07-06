import os
import shutil

import pandas as pd
from tqdm import tqdm

train_df_full = pd.read_csv("~/stanford-rna-3d-folding/train_sequences.csv")

subset_fraction = (
    0.40  # This will give 30% of the original data with some buffer for failed samples
)
sampled_df = train_df_full.sample(frac=subset_fraction, random_state=42).reset_index(
    drop=True
)
train_fraction = 1 / 3
test_fraction = 0.5  # Of the remaining 2/3, take half for test


train_df = sampled_df.sample(frac=train_fraction, random_state=42)
remaining_df = sampled_df.drop(train_df.index)

test_df = remaining_df.sample(frac=test_fraction, random_state=42)
validation_df = remaining_df.drop(
    test_df.index
)  # The rest is for validation (10% of original)

print(len(train_df), len(test_df), len(validation_df))


base_dir = os.path.join("/home/ubuntu/stanford-rna-3d-folding/")
assert os.path.exists(os.path.join(base_dir, "cifs"))
os.makedirs(os.path.join(base_dir, "train_cif"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "test_cif"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "val_cif"), exist_ok=True)

for i, row in tqdm(train_df.iterrows()):
    try:
        shutil.copy(
            os.path.join(
                base_dir, "cifs", (row["target_id"]).lower().split("_")[0] + ".cif"
            ),
            os.path.join(base_dir, "train_cif"),
        )
    except FileNotFoundError:
        print(f"File not found for {row['target_id']}, skipping...")

for i, row in tqdm(test_df.iterrows()):
    try:
        shutil.copy(
            os.path.join(
                base_dir, "cifs", (row["target_id"]).lower().split("_")[0] + ".cif"
            ),
            os.path.join(base_dir, "test_cif"),
        )
    except FileNotFoundError:
        print(f"File not found for {row['target_id']}, skipping...")

for i, row in tqdm(validation_df.iterrows()):
    try:
        shutil.copy(
            os.path.join(
                base_dir, "cifs", (row["target_id"]).lower().split("_")[0] + ".cif"
            ),
            os.path.join(base_dir, "val_cif"),
        )
    except FileNotFoundError:
        print(f"File not found for {row['target_id']}, skipping...")
