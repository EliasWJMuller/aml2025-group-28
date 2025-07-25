import os
import argparse
import torch
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
import random
import numpy as np


from grapharna import dot_to_bpseq, process_rna_file
from grapharna.datasets import RNAPDBDataset
from grapharna.utils import read_dotseq_file
from grapharna.utils.sampler_ddim import Sampler
from grapharna.main_rna_pdb_single_ddim import sample
from grapharna.models import PAMNet, Config


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    seed_everything(seed)
    torch.backends.mps.deterministic = True
    torch.backends.mps.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default=None, help="Input file in *.dotseq format"
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU number.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to be used")
    parser.add_argument(
        "--epochs", type=int, default=150, help="Number of epochs to train."
    )
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate.")
    parser.add_argument(
        "--n_layer", type=int, default=6, help="Number of hidden layers."
    )
    parser.add_argument(
        "--dim", type=int, default=256, help="Size of input hidden units."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="batch_size")
    parser.add_argument(
        "--cutoff_l", type=float, default=0.5, help="cutoff in local layer"
    )
    parser.add_argument(
        "--cutoff_g", type=float, default=1.6, help="cutoff in global layer"
    )
    parser.add_argument("--timesteps", type=int, default=5000, help="timesteps")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument(
        "--mode", type=str, default="coarse-grain", help="Mode of the dataset"
    )
    parser.add_argument("--knns", type=int, default=20, help="Number of knns")
    parser.add_argument(
        "--blocks", type=int, default=6, help="Number of transformer blocks"
    )
    parser.add_argument(
        "--sampling-resids",
        type=str,
        default=None,
        help="Residues that will be sampled, while the rest of the structure will remain fixed",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of the model to be used for sampling",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=800,
        help="Epoch of the model to be used for sampling",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=None,
        help="Number of DDIM steps to use for sampling. If not provided, the full number of timesteps will be used.",
    )
    # parser.add_argument('--fixed-ps', action='store_true', help='If True, P atoms will be fixed and the rest of the structure will be generated. Otherwise, the whole structure will be generated')
    args = parser.parse_args()

    print("Seed:", args.seed)
    set_seed(args.seed)
    # Load the model
    exp_name = args.name if args.name is not None else "grapharna"
    epoch = args.epoch if args.epoch is not None else 800
    model_path = f"save/{exp_name}/model_{epoch}.h5"

    if args.input is None and args.dataset is None:
        # print help message
        print("Please provide input file (or dataset name).")
        return
    elif args.input is not None and args.dataset is not None:
        print("Please provide only one of the following: input file or dataset name.")
        return

    if args.input is not None:
        # generate input file
        print(args.input)
        _, dot, seq = read_dotseq_file(args.input)
        name = os.path.basename(args.input)
        dir_name = name.replace(".dotseq", "")
        bpseq = dot_to_bpseq(dot)
        process_rna_file(
            rna_file=args.input,
            seq_segments=seq,
            file_3d_type=".dotseq",
            sampling=True,
            save_dir_full=f"data/user_inputs/{dir_name}",
            name=name,
            res_pairs=bpseq,
        )
        print(f"Input file generated at! data/user_inputs/{dir_name}")

    config = Config(
        dataset=args.dataset,
        dim=args.dim,
        n_layer=args.n_layer,
        cutoff_l=args.cutoff_l,
        cutoff_g=args.cutoff_g,
        mode=args.mode,
        knns=args.knns,
        transformer_blocks=args.blocks,
    )

    model = PAMNet(config)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device: ", device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    print("Model loaded! WORKS")
    model.eval()
    model.to(device)
    if args.dataset is not None:
        root = os.path.dirname(args.dataset)
        name = os.path.basename(args.dataset)
        ds = RNAPDBDataset(root, name=name, mode=args.mode)
        print(args.batch_size, "batch size")
        print(len(ds), "samples in the dataset")
    else:  # args.input is not None
        name = os.path.basename(args.input)
        dir_name = name.replace(".dotseq", "")
        ds = RNAPDBDataset("data/user_inputs/", name=dir_name, mode=args.mode)

    ds_loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False, pin_memory=False
    )
    print(len(ds_loader), "batches in the dataset")
    sampler = Sampler(timesteps=args.timesteps)
    print("Sampling...")
    sample(
        model,
        ds_loader,
        device,
        sampler,
        epoch,
        args=args,
        num_batches=None,
        exp_name=f"{exp_name}-seed={args.seed}-ddim={args.ddim_steps}",
    )
    print(f"Results stored in path: samples/{exp_name}")


if __name__ == "__main__":
    main()
