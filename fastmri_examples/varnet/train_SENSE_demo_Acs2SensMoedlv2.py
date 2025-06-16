import sys
sys.path.append('/home/jupyter-wangyuwan/wangyuwan/fastmri--varnet/JNU_SZTU_PROJECT/ZGRAPPA_model_work/fastMRI')
import os
import pathlib
from argparse import ArgumentParser

import pytorch_lightning as pl

from torch.serialization import add_safe_globals
from fastmrinew.data.mri_data import fetch_dir
from fastmrinew.data.subsample import create_mask_for_mask_type
from fastmrinew.data.transforms_acs_SENSE import SENSEDataTransform
from fastmrinew.pl_modules import FastMriDataModuleSENSE as FastMriDataModule, SENSEModule

add_safe_globals([pathlib.PosixPath])

def cli_main(args):
    pl.seed_everything(args.seed)

    # ------------
    # data
    # ------------
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    train_transform = SENSEDataTransform(args.racc, mask_func=mask, use_seed=False)
    val_transform = SENSEDataTransform(args.racc, mask_func=mask)
    test_transform = SENSEDataTransform(args.racc)

    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=(args.accelerator in ("ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
    model = SENSEModule(
        racc=args.racc,
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    # ------------
    # trainer
    # ------------
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        deterministic=args.deterministic,
        default_root_dir=args.default_root_dir,
        max_epochs=args.max_epochs,
        callbacks=args.callbacks,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=True,
    )

    # ------------
    # run
    # ------------
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module, ckpt_path=args.ckpt_path)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module, ckpt_path=args.ckpt_path)
    else:
        raise ValueError(f"unrecognized mode {args.mode}")

def build_args():
    parser = ArgumentParser()

    # basic args
    path_config = pathlib.Path("../../fastmri_dirs.yaml")
    backend = "ddp"
    batch_size = 1
    

    data_path = fetch_dir("knee_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "sense_train_mse_loss" / "sense_demo"

    parser.add_argument("--mode", default="train", choices=("train", "test"), type=str)
    parser.add_argument("--racc", required=True, type=int)
    parser.add_argument("--mask_type", choices=("random", "equispaced_fraction"), default="equispaced_fraction", type=str)
    parser.add_argument("--center_fractions", nargs="+", default=[0.15625], type=float)
    parser.add_argument("--accelerations", nargs="+", default=[3], type=int)

    # data config
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,
        mask_type="equispaced_fraction",
        challenge="multicoil",
        batch_size=batch_size,
        test_path=None,
    )

    # module config
    parser = SENSEModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=8,
        pools=4,
        chans=18,
        sens_pools=4,
        sens_chans=8,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
    )

    # Lightning v2.x Trainer config
    parser.add_argument("--accelerator", default="gpu", type=str, help="cpu or gpu")
    parser.add_argument("--devices", default=1, type=int, help="Number of devices (GPUs/CPUs)")
    parser.add_argument("--strategy", default="ddp", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--deterministic", default=True, type=bool)
    parser.add_argument("--default_root_dir", default=default_root_dir, type=pathlib.Path)
    parser.add_argument("--max_epochs", default=50, type=int)
    parser.add_argument("--log_every_n_steps", default=50, type=int)
    parser.add_argument("--ckpt_path", default=None, type=str, help="Checkpoint path for resume")

    args = parser.parse_args()

    # configure checkpointing in checkpoint_dir
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)

    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=1,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]

    # set default checkpoint if one exists in our checkpoint directory
    if args.ckpt_path is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.ckpt_path = str(ckpt_list[-1])

    return args

def run_cli():
    args = build_args()
    cli_main(args)

if __name__ == "__main__":
    run_cli()