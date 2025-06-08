#!/usr/bin/env python
# train_varnet.py

import os
import pathlib
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Sequence, Union

import pytorch_lightning as pl

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
import sys
sys.path.append("./")
from myfastmri.all import FastMriDataModuleV2 as FastMriDataModule
from myfastmri.all import ScaledVarNetDataTransform
from myfastmri.all import VarNetModuleV2 as VarNetModule


def cli_main(args):
    pl.seed_everything(args.seed)

    # ---- Data / Transforms ----
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )

    train_transform = ScaledVarNetDataTransform(mask_func=mask, use_seed=False, scale=args.scale)
    val_transform   = ScaledVarNetDataTransform(mask_func=mask, scale=args.scale)
    test_transform  = ScaledVarNetDataTransform(mask_func=mask, scale=args.scale)

    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=args.sample_rate,
        volume_sample_rate=args.volume_sample_rate,
        val_sample_rate=args.val_sample_rate,
        val_volume_sample_rate=args.val_volume_sample_rate,
        test_sample_rate=args.test_sample_rate,
        test_volume_sample_rate=args.test_volume_sample_rate,
        use_dataset_cache_file=args.use_dataset_cache_file,
        combine_train_val=args.combine_train_val,
        test_split=args.test_split,
        test_path=args.test_path,
        distributed_sampler=(args.strategy in ("ddp", "ddp_spawn")),
    )

    # ---- Model ----
    model = VarNetModule(
        num_cascades=args.num_cascades,
        pools=args.pools,
        chans=args.chans,
        sens_pools=args.sens_pools,
        sens_chans=args.sens_chans,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
    )

    # ---- Trainer ----
    trainer = pl.Trainer(
        accelerator=args.accelerator,       # "cpu" or "gpu"
        devices=args.devices,               # number of GPUs (or list)
        strategy=args.strategy,             # e.g. "ddp_spawn"
        precision=args.precision,           # 16 for mixed precision
        benchmark=args.benchmark,           # cudnn.benchmark
        default_root_dir=args.default_root_dir,
        max_epochs=args.max_epochs,
        callbacks=args.callbacks,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=True,
    )

    # ---- Run ----
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    elif args.mode == "test":
        trainer.test(model, datamodule=data_module)
    else:
        raise ValueError(f"Unknown mode {args.mode}")


def build_args():
    parser = ArgumentParser()

    # basic
    path_config   = pathlib.Path("../../fastmri_dirs.yaml")
    backend       = "ddp"
    batch_size    = 1

    data_path     = fetch_dir("knee_path", path_config)
    default_root  = fetch_dir("log_path", path_config) / "varnet" / "lightning2"

    # mode
    parser.add_argument("--mode", default="train", choices=("train","test"))

    # mask
    parser.add_argument("--mask_type", choices=("random","equispaced_fraction"), default="equispaced_fraction")
    parser.add_argument("--center_fractions", nargs="+", type=float, default=[0.08])
    parser.add_argument("--accelerations",   nargs="+", type=int,   default=[4])

    # data args
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor to multiply k-space and target images (use >1.0 for FP16)",
    )
    parser.set_defaults(
        data_path=data_path,
        challenge="multicoil",
        batch_size=batch_size,
        num_workers=1,
        sample_rate=None,
        volume_sample_rate=None,
        val_sample_rate=None,
        val_volume_sample_rate=None,
        test_sample_rate=None,
        test_volume_sample_rate=None,
        use_dataset_cache_file=True,
        combine_train_val=False,
        test_split="test",
        test_path=None,
    )

    # model args
    parser = VarNetModule.add_model_specific_args(parser)
    parser.set_defaults(
        num_cascades=1,
        pools=4,
        chans=18,
        sens_pools=4,
        sens_chans=8,
        lr=1e-3,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
    )

    # trainer args (Lightning 2.x style)
    parser.add_argument("--accelerator", choices=("cpu","gpu"), default="gpu")
    parser.add_argument("--devices", type=int, default=2)
    parser.add_argument("--strategy",    choices=["ddp","ddp_spawn"], default="ddp")
    # "16-mixed" is not suitable to handle raw fastmri brain multicoil data, which has very small values
    # "32-true" is recommended for this data, unless dataset is normalized
    parser.add_argument("--precision",   choices=["32-true","16-mixed"], default="32-true") 
    parser.add_argument("--benchmark",   action="store_true")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.set_defaults(
        seed=42,
        deterministic=True,
        default_root_dir=default_root,
        max_epochs=200,
    )

    # checkpoint callback
    args = parser.parse_args()
    cb = pl.callbacks.ModelCheckpoint(
        dirpath=Path(args.default_root_dir)/"checkpoints",
        filename="varnet-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        monitor="validation_loss",
        mode="min",
        every_n_epochs=1,
    )
    args.callbacks = [cb]

    return args


def run_cli():
    args = build_args()
    os.makedirs(args.default_root_dir, exist_ok=True)
    run_args = vars(args).copy()
    cli_main(args)


if __name__ == "__main__":
    run_cli()
