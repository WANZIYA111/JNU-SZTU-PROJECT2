"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import sys
sys.path.append('/home/jupyter-wangyuwan/wangyuwan/fastmri--varnet/JNU_SZTU_PROJECT/ZGRAPPA_model_work/fastMRI')

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from fastmrinew.models.fast_sense import SENSE

import fastmri
from fastmri.data import transforms


from .varnet import NormUnet


class SensitivityModelSENSE(nn.Module):
    """
    Model for learning sensitivity estimation from k-space data.

    This model applies an IFFT to multichannel k-space data and then a U-Net
    to the coil images to estimate coil sensitivities. It can be used with the
    end-to-end variational network.
    """

    def __init__(
        self,
        chans: int,
        num_pools: int,
        in_chans: int = 2,
        out_chans: int = 2,
        drop_prob: float = 0.0,
        mask_center: bool = True,
    ):
       
        """
        Args:
            chans: Number of output channels of the first convolution layer.
            num_pools: Number of down-sampling and up-sampling layers.
            in_chans: Number of channels in the input to the U-Net model.
            out_chans: Number of channels in the output to the U-Net model.
            drop_prob: Dropout probability.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()
        self.mask_center = mask_center
        self.norm_unet = NormUnet(
            chans,
            num_pools,
            in_chans=in_chans,
            out_chans=out_chans,
            drop_prob=drop_prob,
        )

    def chans_to_batch_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        b, c, h, w, comp = x.shape

        return x.view(b * c, 1, h, w, comp), b

    def batch_chans_to_chan_dim(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        bc, _, h, w, comp = x.shape
        c = bc // batch_size

        return x.view(batch_size, c, h, w, comp)

    def divide_root_sum_of_squares(self, x: torch.Tensor) -> torch.Tensor:
        return x / fastmri.rss_complex(x, dim=1).unsqueeze(-1).unsqueeze(1)

    def get_pad_and_num_low_freqs(
        self, mask: torch.Tensor, num_low_frequencies: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_low_frequencies is None or num_low_frequencies == 0:
            # get low frequency line locations and mask them out
            squeezed_mask = mask[:, 0, 0, :, 0].to(torch.int8)
            cent = squeezed_mask.shape[1] // 2
            # running argmin returns the first non-zero
            left = torch.argmin(squeezed_mask[:, :cent].flip(1), dim=1)
            right = torch.argmin(squeezed_mask[:, cent:], dim=1)
            num_low_frequencies_tensor = torch.max(
                2 * torch.min(left, right), torch.ones_like(left)
            )  # force a symmetric center unless 1
        else:
            num_low_frequencies_tensor = num_low_frequencies * torch.ones(
                mask.shape[0], dtype=mask.dtype, device=mask.device
            )

        pad = (mask.shape[-2] - num_low_frequencies_tensor + 1) // 2

        return pad.type(torch.long), num_low_frequencies_tensor.type(torch.long)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        device = masked_kspace.device
        if self.mask_center:
            pad, num_low_freqs = self.get_pad_and_num_low_freqs(
                mask, num_low_frequencies
            )
            masked_kspace = transforms.batched_mask_center(
                masked_kspace, pad, pad + num_low_freqs
            )
        pd = 30
        ACS_MASK = torch.zeros(masked_kspace.shape).to(device)
        h = int(masked_kspace.shape[-3])
        w = int(masked_kspace.shape[-2])
        ACS_MASK[...,int(h/2)-pd:int(h/2)+pd,int(w/2)-pd:int(w/2)+pd,:] = 1
        ACS_kspace = masked_kspace*ACS_MASK
        # convert to image space
        images, batches = self.chans_to_batch_dim(fastmri.ifft2c(ACS_kspace.to(device)))
        # np.save('SENSE_ACS_kspace',torch.view_as_complex(ACS_kspace).detach().cpu().numpy())
        del masked_kspace, mask,ACS_MASK
        # estimate sensitivities
        return self.divide_root_sum_of_squares(
            self.batch_chans_to_chan_dim(self.norm_unet(images), batches)
        ),ACS_kspace


class SENSEModel(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
        self,
        racc:int,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
        """
        super().__init__()

        self.sens_net = SensitivityModelSENSE(
            chans=sens_chans,
            num_pools=sens_pools,
            mask_center=mask_center,
        )
        self.racc = racc

    def forward(
        self,
        masked_kspace: torch.Tensor,
        real_masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        real_mask: torch.Tensor,
        num_low_frequencies: Optional[int] = None,
    ) -> torch.Tensor:
        device = real_masked_kspace.device
        sens_maps,ACS_kspace = self.sens_net(masked_kspace, mask, num_low_frequencies)
        R = self.racc
        # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA:",R)
        acc_factor = torch.tensor(R)
        sense_kspace = torch.view_as_complex(real_masked_kspace).permute(0,2,3,1)
        
        sense_sens = torch.view_as_complex(sens_maps).permute(0,2,3,1)
        image = SENSE(sense_kspace,sense_sens,acc_factor)#image [batch,x,y]
        ## 输入inp和csm形状应为 [batch, kx, ky, coil]
        return abs(image).float(),sens_maps,ACS_kspace,real_masked_kspace


