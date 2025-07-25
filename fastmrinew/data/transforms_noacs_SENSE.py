from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch

import fastmrinew

from .subsample import MaskFunc
from .transforms import to_tensor,apply_mask,batched_mask_center,center_crop

class SENSESample_noacs(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum image value.
        crop_size: The size to crop the final image.
    """

    masked_kspace: torch.Tensor
    real_masked_kspace: torch.Tensor
    acs_kspace:torch.Tensor
    gold_sens:torch.Tensor
    mask: torch.Tensor
    real_mask: torch.Tensor
    # weight_mask: torch.Tensor
    num_low_frequencies: Optional[int]
    target: torch.Tensor
    fname: str
    slice_num: int
    max_value: float
    crop_size: Tuple[int, int]

class SENSEDataTransform_noacs:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(self, racc: int, mask_func: Optional[MaskFunc] = None, use_seed: bool = True):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.racc = racc

    def __call__(
        self,
        kspace: np.ndarray,
        ACS_KSPACE : np.ndarray,
        gold_sens : np.ndarray,
        mask: np.ndarray,
        target: Optional[np.ndarray],
        # sens_weight_mask: np.ndarray,
        attrs: Dict,
        fname: str,
        slice_num: int,
    ) -> SENSESample_noacs:
        # print("fname:",fname,"slice_num:",slice_num)
        """
        Args:
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            mask: Mask from the test dataset.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice_num: Serial number of the slice.

        Returns:
            A VarNetSample with the masked k-space, sampling mask, target
            image, the filename, the slice number, the maximum image value
            (from target), the target crop size, and the number of low
            frequency lines sampled.
        """
        if target is not None:
            target_torch = to_tensor(target)
            max_value = attrs["max"]
        else:
            target_torch = torch.tensor(0)
            max_value = 0.0

        kspace_torch = to_tensor(kspace)
        acs_kspace = to_tensor(ACS_KSPACE)
        device = kspace_torch.device
        # kspace_torch torch.Size([16, 384, 384, 2])
        
        '''
        Code add by wangyuwan
        '''
        RACC =self.racc
        real_mask = torch.zeros(kspace_torch.shape).to(device)
        real_mask[:,(slice_num%RACC)::RACC,:,:] = 1
        real_masked_kspace = kspace_torch*real_mask
        
        
        seed = None if not self.use_seed else tuple(map(ord, fname))
        acq_start = attrs["padding_left"]
        acq_end = attrs["padding_right"]

        crop_size = (attrs["recon_size"][0], attrs["recon_size"][1])

        if self.mask_func is not None:
            masked_kspace, mask_torch, num_low_frequencies = apply_mask(
                kspace_torch, self.mask_func, seed=seed, padding=(acq_start, acq_end)
            )

            sample = SENSESample_noacs(
                masked_kspace=masked_kspace,
                real_masked_kspace = real_masked_kspace,
                acs_kspace = acs_kspace,
                gold_sens = gold_sens,
                mask=mask_torch.to(torch.bool),
                real_mask = real_mask.to(torch.bool),
                # weight_mask = torch.from_numpy(sens_weight_mask).to(torch.bool),
                num_low_frequencies=num_low_frequencies,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
            )
            # np.save('mask.npy',mask_torch.detach().cpu().numpy())(1, 1, 384, 1)
            # np.save('masked_kspace.npy',masked_kspace.detach().cpu().numpy())(16, 384, 384, 2)
        else:
            masked_kspace = kspace_torch
            shape = np.array(kspace_torch.shape)
            num_cols = shape[-2]
            shape[:-3] = 1
            mask_shape = [1] * len(shape)
            mask_shape[-2] = num_cols
            mask_torch = torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))
            mask_torch = mask_torch.reshape(*mask_shape)
            mask_torch[:, :, :acq_start] = 0
            mask_torch[:, :, acq_end:] = 0

            sample = VarNetSample(
                masked_kspace=masked_kspace.to(torch.bool),
                mask=mask_torch.to(torch.bool),
                num_low_frequencies=0,
                target=target_torch,
                fname=fname,
                slice_num=slice_num,
                max_value=max_value,
                crop_size=crop_size,
            )

        return sample