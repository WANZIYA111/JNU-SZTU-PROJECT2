import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def SENSE(inp, csm, acc_factor, replicas=100, samp_mat=None, reg=None):
    """
    inp: [batch, x, y, coils] complex tensor, undersampled k-space
    csm: [batch, x, y, coils] complex tensor, coil sensitivity map
    acc_factor: int or tuple/list, acceleration factor(s)
    replicas: int, number of pseudo-replica runs (default 100)
    samp_mat: [batch, x, y] binary tensor, sampling mask (optional)
    reg: float, regularization factor (optional)
    Returns:
        img: [batch, x, y] complex tensor, SENSE reconstructed image
        img_alias: [batch, x, y, coils] complex tensor
    """
    assert inp.ndim == 4 and csm.ndim == 4, "inp and csm must be [batch, x, y, coil]"
    batch_size = inp.shape[0]

    if samp_mat is None:
        samp_mat = (inp.abs().sum(dim=3) > 0).to(inp.dtype)  # [batch, x, y]

    if reg is None:
        reg = 0.0

    imgs = []
    for b in range(batch_size):
        # Calculate SENSE unmixing coefficients
        unmix_sense = ismrm_calculate_sense_unmixing_general2(acc_factor, csm[b], None, reg)
        # SENSE image reconstruction
        img_alias = torch.sqrt(torch.prod(torch.tensor(acc_factor))) * ismrm_transform_kspace_to_image(inp[b], [0, 1])
        img = torch.sum(img_alias * unmix_sense, dim=2)
        imgs.append(img)
    imgs = torch.stack(imgs, dim=0)  # [batch, x, y]
    return imgs

def ismrm_transform_kspace_to_image(k, dim=[0,1], img_shape=None):
    if img_shape is None:
        img_shape = k.shape 
    else:
        img_shape = tuple(img_shape)
    
    img = k.clone()
    for d in dim:
        
        img = torch.fft.ifftshift(img, dim=d) 
        img = torch.fft.ifft(img, n=img_shape[d], dim=d, norm="ortho")
        img = torch.fft.fftshift(img, dim=d)
    
    return img
def ismrm_calculate_sense_unmixing_general2(acc_factor, csm, noise_matrix=None, regularization_factor=None):
    """
    acc_factor: int or tuple/list of 2 ints
    csm: torch.complex64 tensor, shape [x, y, coil]
    noise_matrix: torch.complex64 tensor, shape [coil, coil] (optional)
    regularization_factor: float (optional)
    Returns:
        unmix: torch.complex64 tensor, shape [x, y, coil]
        gmap: torch.float32 tensor, shape [x, y]
    """
    assert csm.ndim == 3, "coil sensitivity map must have 3 dimensions"
    nx, ny, nc = csm.shape

    if noise_matrix is None:
        noise_matrix = torch.eye(nc, dtype=csm.dtype, device=csm.device)
    if regularization_factor is None:
        regularization_factor = 0.0
    
    noise_matrix_inv = torch.linalg.pinv(noise_matrix)

    if len(acc_factor) == 1:
        # 1D SENSE
        unmix = torch.zeros_like(csm)
        for x in range(nx):
            csm1d = csm[x, :, :]  # shape [ny, nc]
            unmix1d = ismrm_calculate_sense_unmixing_1d(acc_factor, csm1d, noise_matrix_inv, regularization_factor)
            unmix[x, :, :] = unmix1d
    elif len(acc_factor) == 2:
        
        unmix = ismrm_calculate_sense_unmixing_2d_actual(acc_factor, csm, noise_matrix_inv, regularization_factor)
    else:
        raise ValueError('acceleration factor cannot have more than 3 elements')

   
    return unmix



def ismrm_calculate_sense_unmixing_2d_actual(acc_factor, csm2d, noise_matrix_inv, regularization_factor):
    """
    acc_factor: tuple/list of 2 ints
    csm2d: [nx, ny, nc]
    """
    nx, ny, nc = csm2d.shape
    acc_factor_x, acc_factor_y = acc_factor
    if nx % acc_factor_x != 0:
        raise ValueError('nx must be a multiple of acc_factor(0)')
    if ny % acc_factor_y != 0:
        raise ValueError('ny must be a multiple of acc_factor(1)')

    unmix2d = torch.zeros_like(csm2d)
    nx_blocks = nx // acc_factor_x
    ny_blocks = ny // acc_factor_y

    def process_block(args):
        ix_blocks, iy_blocks = args
        sub_csm = csm2d[ix_blocks::nx_blocks, iy_blocks::ny_blocks, :]
        A = sub_csm.reshape(acc_factor_x*acc_factor_y,-1)
        A = A.T
        block = torch.zeros(acc_factor_x, acc_factor_y, nc, dtype=A.dtype, device=A.device)
        if torch.max(torch.abs(A)) > 0.01:
            pixels_unaliased = torch.zeros(A.shape[1], A.shape[0], dtype=A.dtype, device=A.device)
            nonzeros_index = (torch.sum(torch.abs(A), dim=0) > 0.01)
            A_reduced = A[:, nonzeros_index]
            AHA = A_reduced.conj().T @ A_reduced
            diag_mask = torch.abs(torch.diag(AHA)) > 0.01
            reduced_eye = torch.diag(diag_mask.to(AHA.dtype))
            n_alias = reduced_eye.sum().item()
            scaled_reg_factor = regularization_factor * torch.trace(AHA) / n_alias 
            X = torch.linalg.solve(AHA + reduced_eye * scaled_reg_factor, A_reduced.conj().T)
            X_reshaped = X.reshape(A_reduced.shape[1], nc)
            pixels_unaliased[nonzeros_index, :] = X_reshaped
            block = pixels_unaliased.reshape(acc_factor_x, acc_factor_y, nc)
        return (ix_blocks, iy_blocks, block)

    block_args = [(ix_blocks, iy_blocks) for ix_blocks in range(nx_blocks) for iy_blocks in range(ny_blocks)]
    results = []
    with ThreadPoolExecutor() as executor:
        for res in executor.map(process_block, block_args):
            results.append(res)
    for ix_blocks, iy_blocks, block in results:
        unmix2d[ix_blocks::nx_blocks, iy_blocks::ny_blocks, :] = block
    return unmix2d