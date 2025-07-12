import sys
import os
import numpy as np
from Bio.PDB import PDBParser
from tmtools import tm_align

# Define 3-letter to 1-letter conversion for amino acids
PROTEIN_LETTERS_3_TO_1 = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
                          'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
                          'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
                          'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

# Define a mapping for nucleic acid bases
NUCLEIC_ACID_LETTERS = {"A": "A", "C": "C", "G": "G", "T": "T", "U": "U"}

def get_structure_from_pdb(pdb_path):
    """Parses a PDB file and returns a structure object."""
    parser = PDBParser(QUIET=True)
    return parser.get_structure("structure", pdb_path)

def get_coords_and_seq(pdb_path):
    """
    Reads coordinates and sequence from the first valid chain in a PDB file.
    Calls the universal get_residue_data function.
    """
    structure = get_structure_from_pdb(pdb_path)
    for chain in structure.get_chains():
        try:
            coords, seq = get_residue_data(chain)
            # If data is extracted, print info and return it
            if coords.shape[0] > 0:
                print(f"✔️ Successfully processed chain {chain.id} from '{pdb_path}'")
                return coords, seq
        except Exception as e:
            print(f"⚠️ Warning: Could not process chain {chain.id} in '{pdb_path}'. Reason: {e}")
            continue
    # If no valid chain is found after checking all, raise an error
    raise ValueError(f"Could not extract residue data from any chain in {pdb_path}")

def get_residue_data(chain, ignore_hetero=True):
    """
    Extracts backbone coordinates and sequence from a PDB chain.
    Automatically detects if the chain is a Protein (CA) or RNA/DNA (C4').
    """
    coords = []
    seq = []
    mode = None

    # Sniff the first few residues to determine if it's protein or nucleic acid
    test_residues = list(chain.get_residues())[:10]
    if any("CA" in r for r in test_residues):
        mode = "protein"
    elif any("C4'" in r for r in test_residues):
        mode = "nucleic"
    else: # If we can't determine the type, we can't process it
        return np.empty((0, 3)), ""

    for residue in chain.get_residues():
        # Skip non-standard residues like water molecules (HETATMs)
        if residue.id[0] != ' ' and ignore_hetero:
            continue

        if mode == "protein" and "CA" in residue:
            coords.append(residue["CA"].coord)
            seq.append(PROTEIN_LETTERS_3_TO_1.get(residue.resname, 'X')) # 'X' for unknown amino acids
        
        elif mode == "nucleic" and "C4'" in residue:
            coords.append(residue["C4'"].coord)
            # For PDBs, the base name is usually the last character (e.g., '  G' -> 'G')
            base = residue.resname.strip()[-1]
            seq.append(NUCLEIC_ACID_LETTERS.get(base, 'N')) # 'N' for unknown bases

    if not coords:
        return np.empty((0, 3)), ""

    return np.vstack(coords), "".join(seq)

def main():
    """
    Calculates the TM-score for all predicted structures against their targets.
    """
    if len(sys.argv) != 3:
        print("Usage: python score_dir.py <preds_dir> <target_dir>", file=sys.stderr)
        sys.exit(1)

    preds_dir = sys.argv[1]
    target_dir = sys.argv[2]

    if not os.path.isdir(preds_dir):
        print(f"❌ Error: Prediction directory not found at '{preds_dir}'", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isdir(target_dir):
        print(f"❌ Error: Target directory not found at '{target_dir}'", file=sys.stderr)
        sys.exit(1)

    tm_scores = []
    
    # Assuming filenames in preds_dir and target_dir match
    for pred_filename in os.listdir(preds_dir):
        try:
            if not pred_filename.endswith((".pdb", ".cif")): # Basic filter for PDB/CIF files
                continue

            pred_filepath = os.path.join(preds_dir, pred_filename)
            target_filepath = os.path.join(target_dir, pred_filename)

            if not os.path.exists(target_filepath):
                print(f"⚠️ Warning: Target file not found for '{pred_filename}' in '{target_dir}'. Skipping.")
                continue

            # Prediction is chain1, Target is chain2
            coords1, seq1 = get_coords_and_seq(pred_filepath)
            coords2, seq2 = get_coords_and_seq(target_filepath)
            
            if coords1.shape[0] == 0 or coords2.shape[0] == 0:
                print(f"⚠️ Warning: Could not get coordinates for '{pred_filename}'. Skipping.")
                continue

            res = tm_align(coords1, coords2, seq1, seq2)
            
            # We are interested in the TM-score normalized by the length of the target structure (chain2)
            tm_scores.append(res.tm_norm_chain2)
            print(f"Processed '{pred_filename}': TM-score = {res.tm_norm_chain2:.4f}")

        except (FileNotFoundError, ValueError, IndexError, RuntimeError) as e:
            print(f"❌ Error processing '{pred_filename}': {e}", file=sys.stderr)
            continue
    
    if tm_scores:
        mean_tm = np.mean(tm_scores)
        std_tm = np.std(tm_scores)
        
        print("-" * 40)
        print(f"TM-Score Summary:")
        print(f"  Processed {len(tm_scores)} structure pairs.")
        print(f"  Mean: {mean_tm:.4f}")
        print(f"  Standard Deviation: {std_tm:.4f}")
        print("-" * 40)
    else:
        print("No structure pairs were processed.")


if __name__ == "__main__":
    main()
