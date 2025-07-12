import argparse
import numpy as np

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


def read_sequence_from_file(filepath):
    """
    Reads an RNA sequence from a .dotseq or FASTA-like file.
    """
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(tuple(">;#")):
                continue
            if all(c in "ACGUacgu" for c in line):
                return line.upper()
    raise ValueError(f"No valid RNA sequence found in {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate baseline RNA 3D structure")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-s", "--seq", type=str, help="RNA sequence (e.g. GAUAAGGCC)"
    )
    input_group.add_argument(
        "-i",
        "--infile",
        type=str,
        help="Input file (.dotseq, .fasta) containing RNA sequence",
    )
    parser.add_argument(
        "-o", "--out", type=str, default="rna_baseline.pdb", help="Output PDB filename"
    )
    # Restoring the command-line arguments for geometric parameters
    parser.add_argument(
        "--bond-length", type=float, default=3.8, help="Fixed bond length in Angstroms"
    )
    parser.add_argument(
        "--bond-angle-range",
        nargs=2,
        type=float,
        default=[90, 150],
        help="Min and max bond angle in degrees",
    )
    parser.add_argument(
        "--dihedral-range",
        nargs=2,
        type=float,
        default=[-180, 180],
        help="Min and max dihedral angle in degrees",
    )
    args = parser.parse_args()

    if args.seq:
        seq = args.seq.strip().upper()
    else:
        seq = read_sequence_from_file(args.infile)

    # Pass the arguments from the command line to the coordinate generation function
    coords = generate_coords(
        seq,
        bond_length=args.bond_length,
        bond_angle_range=tuple(args.bond_angle_range),
        dihedral_range=tuple(args.dihedral_range),
    )
    write_pdb(seq, coords, args.out)


if __name__ == "__main__":
    main()
