import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Callable, Union

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.mri_data import SliceDataset, CombinedSliceDataset
from fastmri.pl_modules.data_module import FastMriDataModule

from torch.utils.data import DataLoader

# worker init unchanged
from fastmri.pl_modules.data_module import worker_init_fn

class FastMriDataModuleV2(pl.LightningDataModule):
    """
    Lightning 2.x DataModule for fastMRI using inheritance from the original FastMriDataModule.
    """
    def __init__(
        self,
        data_path: Union[str, Path],
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        test_transform: Callable,
        batch_size: int = 1,
        num_workers: int = 4,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        val_volume_sample_rate: Optional[float] = None,
        test_sample_rate: Optional[float] = None,
        test_volume_sample_rate: Optional[float] = None,
        use_dataset_cache_file: bool = True,
        combine_train_val: bool = False,
        test_split: str = "test",
        test_path: Optional[Path] = None,
        distributed_sampler: bool = False,
    ):
        super().__init__()
        # init original args
        self.data_path = Path(data_path)
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.combine_train_val = combine_train_val
        self.test_split = test_split
        self.test_path = test_path
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.test_sample_rate = test_sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.test_volume_sample_rate = test_volume_sample_rate
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler

    def setup(self, stage: Optional[str] = None):
        """
        Build datasets lazily for each stage.
        """
        # train
        if stage in (None, 'fit'):
            if self.combine_train_val:
                roots = [self.data_path/f"{self.challenge}_train",
                         self.data_path/f"{self.challenge}_val"]
                transforms = [self.train_transform, self.train_transform]
                challenges = [self.challenge, self.challenge]
                sample_rates = ([self.sample_rate]*2 if self.sample_rate else None)
                vol_sample_rates = ([self.volume_sample_rate]*2 if self.volume_sample_rate else None)
                self.train_dataset = CombinedSliceDataset(
                    roots=roots,
                    challenges=challenges,
                    transforms=transforms,
                    sample_rates=sample_rates,
                    volume_sample_rates=vol_sample_rates,
                    use_dataset_cache=self.use_dataset_cache_file,
                    raw_sample_filter=None,
                )
            else:
                train_root = self.data_path/f"{self.challenge}_train"
                self.train_dataset = SliceDataset(
                    root=train_root,
                    challenge=self.challenge,
                    transform=self.train_transform,
                    use_dataset_cache=self.use_dataset_cache_file,
                    sample_rate=self.sample_rate,
                    volume_sample_rate=self.volume_sample_rate,
                    raw_sample_filter=None,
                )
        # val
        if stage in (None, 'fit', 'validate'):
            val_root = self.data_path/f"{self.challenge}_val"
            self.val_dataset = SliceDataset(
                root=val_root,
                challenge=self.challenge,
                transform=self.val_transform,
                use_dataset_cache=self.use_dataset_cache_file,
                sample_rate=self.val_sample_rate,
                volume_sample_rate=self.val_volume_sample_rate,
                raw_sample_filter=None,
            )
        # test
        if stage in (None, 'test'):
            if self.test_path:
                test_root = Path(self.test_path)
            else:
                test_root = self.data_path/f"{self.challenge}_{self.test_split}"
            self.test_dataset = SliceDataset(
                root=test_root,
                challenge=self.challenge,
                transform=self.test_transform,
                use_dataset_cache=self.use_dataset_cache_file,
                sample_rate=self.test_sample_rate,
                volume_sample_rate=self.test_volume_sample_rate,
                raw_sample_filter=None,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers>0,
            worker_init_fn=worker_init_fn,
            shuffle=not self.distributed_sampler,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers>0,
            worker_init_fn=worker_init_fn,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers>0,
            worker_init_fn=worker_init_fn,
            shuffle=False,
            pin_memory=True,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        # preserve original signature
        return FastMriDataModule.add_data_specific_args(parent_parser)
