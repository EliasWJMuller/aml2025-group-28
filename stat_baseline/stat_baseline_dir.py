import argparse
import numpy as np
import os
import glob

# Map single-letter to 3-letter PDB residue names for RNA
RES_MAP = {"A": "A", "C": "C", "G": "G", "U": "U"}


def normalize(v):
    """Normalizes a vector to unit length."""
    return v / np.linalg.norm(v)


def random_direction(prev_dir, bond_angle_range, dihedral_range):
    """
    Generate a unit vector at a random bond angle and dihedral relative to prev_dir.
    """
    theta = np.deg2rad(np.random.uniform(*bond_angle_range))
    phi = np.deg2rad(np.random.uniform(*dihedral_range))
    n = normalize(prev_dir)
    if abs(n[0]) < 0.9:
        up = np.array([1.0, 0.0, 0.0])
    else:
        up = np.array([0.0, 1.0, 0.0])
    u = normalize(np.cross(n, up))
    w = np.cross(n, u)
    dir_vec = (
        n * np.cos(theta)
        + u * np.sin(theta) * np.cos(phi)
        + w * np.sin(theta) * np.sin(phi)
    )
    return normalize(dir_vec)


def generate_coords(
    seq, bond_length=3.8, bond_angle_range=(90, 150), dihedral_range=(-180, 180)
):
    """
    Generate 3D coordinates for each nucleotide in seq.
    """
    coords = [np.array([0.0, 0.0, 0.0])]
    prev_dir = np.array([1.0, 0.0, 0.0])
    for i in range(1, len(seq)):
        dir_vec = random_direction(prev_dir, bond_angle_range, dihedral_range)
        new_point = coords[-1] + bond_length * dir_vec
        coords.append(new_point)
        prev_dir = dir_vec
    return coords


def write_pdb(seq, coords, output_file, chain_id="A"):
    """
    Write a simple PDB with one ATOM record per nucleotide.
    This version writes a C4' atom to be compatible with standard tools.
    """
    with open(output_file, "w") as f:
        f.write("HEADER    RNA baseline structure (C4' backbone)\n")
        for i, (base, coord) in enumerate(zip(seq, coords), start=1):
            resname = RES_MAP.get(base.upper(), "N")
            
            # This line creates a C4' atom (element C) to be compatible with the scoring script.
            atom_line = (
                f"ATOM  {i:5d}  C4' {resname:3s} {chain_id}{i:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00           C\n"
            )
            f.write(atom_line)
        f.write("END\n")
    print(f"Wrote PDB to {output_file}")


def read_sequence_from_pdb(filepath):
    """
    Reads an RNA sequence from a PDB file by observing unique residues.
    Maps standard 3-letter residue names to 1-letter codes.
    """
    PDB_RES_TO_ONE = {
        "A": "A", "ADE": "A",
        "C": "C", "CYT": "C",
        "G": "G", "GUA": "G",
        "U": "U", "URA": "U",
    }
    
    seq = []
    seen_residues = set()
    with open(filepath, 'r') as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            
            chain_id = line[21]
            res_seq = int(line[22:26])
            res_name = line[17:20].strip()
            
            residue_id = (chain_id, res_seq)
            
            if residue_id not in seen_residues:
                one_letter_code = PDB_RES_TO_ONE.get(res_name.upper())
                if one_letter_code:
                    seq.append(one_letter_code)
                    seen_residues.add(residue_id)
                else:
                    # This avoids printing warnings for non-standard residues like MET, etc.
                    # that might be in a protein-RNA complex PDB.
                    pass

    return "".join(seq)


def main():
    parser = argparse.ArgumentParser(
        description="Generate random baseline RNA 3D structures based on sequences from a directory of PDB files."
    )
    parser.add_argument(
        "--preds_dir",
        type=str,
        required=True,
        help="Input directory containing PDB files to imitate.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save generated PDB files.",
    )
    parser.add_argument(
        "--bond-length", type=float, default=3.8, help="Fixed bond length in Angstroms."
    )
    parser.add_argument(
        "--bond-angle-range",
        nargs=2,
        type=float,
        default=[90, 150],
        help="Min and max bond angle in degrees.",
    )
    parser.add_argument(
        "--dihedral-range",
        nargs=2,
        type=float,
        default=[-180, 180],
        help="Min and max dihedral angle in degrees.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pdb_files = glob.glob(os.path.join(args.preds_dir, '*.pdb'))
    if not pdb_files:
        print(f"Warning: No .pdb files found in {args.preds_dir}")
        return

    print(f"Found {len(pdb_files)} PDB files to process.")

    for pdb_path in pdb_files:
        base_filename = os.path.basename(pdb_path)
        print(f"Processing {base_filename}...")
        try:
            seq = read_sequence_from_pdb(pdb_path)
            if not seq:
                print(f"Warning: No valid RNA sequence found in {base_filename}. Skipping.")
                continue

            coords = generate_coords(
                seq,
                bond_length=args.bond_length,
                bond_angle_range=tuple(args.bond_angle_range),
                dihedral_range=tuple(args.dihedral_range),
            )
            
            output_path = os.path.join(args.output_dir, base_filename)
            write_pdb(seq, coords, output_path)

        except Exception as e:
            print(f"Error processing {base_filename}: {e}")


if __name__ == "__main__":
    main()