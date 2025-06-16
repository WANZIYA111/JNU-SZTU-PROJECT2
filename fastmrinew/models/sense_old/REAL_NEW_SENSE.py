import torch
import scipy.io
import numpy as np
def SENSE(inp, csm, acc_factor, replicas=100, samp_mat=None, reg=None):
    """
    inp: [kx, ky, coils] complex tensor, undersampled k-space
    csm: [x, y, coils] complex tensor, coil sensitivity map
    acc_factor: int or tuple/list, acceleration factor(s)
    replicas: int, number of pseudo-replica runs (default 100)
    samp_mat: [kx, ky] binary tensor, sampling mask (optional)
    reg: float, regularization factor (optional)
    Returns:
        img: [x, y] complex tensor, SENSE reconstructed image
        gmap: [x, y] real tensor, g-factor map
        snr: [x, y] real tensor, SNR map
        snr_pseudo, gmap_pseudo, noise_psf_pseudo: (optional) pseudo-replica results
    """
    # Prepare sampling matrix
    if samp_mat is None:
        samp_mat = (inp.abs().sum(dim=2) > 0).to(inp.dtype)

    # Regularization
    if reg is None:
        reg = 0.0
    
    # Calculate SENSE unmixing coefficients and gmap
    unmix_sense = ismrm_calculate_sense_unmixing_general2(acc_factor, csm, None, reg)###√肉眼看与matlab的生成矩阵基本上是没有区别的啊。有误差，但是最大的误差在1e-6（大多为1e-8和1e-9），unmix_sense最小值为1.7552e-04

    # SENSE image reconstruction
    img_alias = torch.sqrt(torch.prod(torch.tensor(acc_factor)))*ismrm_transform_kspace_to_image(inp, [0, 1])
    # ceshi0 = torch.sqrt(torch.prod(torch.tensor(acc_factor)))    √
    
   
    img = torch.sum(img_alias * unmix_sense, dim=2)
    

    return img,img_alias

# def ismrm_transform_kspace_to_image(k, dim, img_shape=None):
#     """
#     k: torch.Tensor (complex)
#     dim: list of int, dimensions to transform (e.g., [0,1])
#     img_shape: list/tuple, desired output shape (optional)
#     """
#     if img_shape is None:
#         img_shape = list(k.shape)
#     img = k
#     for i, d in enumerate(dim):
#         img = transform_one_dim(img, d, img_shape[d])
#     return img

# def transform_one_dim(k, dim, img_extent):
#     img = k.clone()
#     # Apply centered ifft along the given dimension
#     img = torch.fft.ifftshift(img, dim=dim)
#     img = torch.fft.ifft(img,dim=dim,norm="ortho")
#     img = torch.fft.fftshift(img, dim=dim)
#     return img
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


import torch

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

# def ismrm_calculate_sense_unmixing_1d(acc_factor, csm1d, noise_matrix_inv, regularization_factor):
#     """
#     csm1d: [ny, nc]
#     """
#     ny, nc = csm1d.shape
#     if ny % acc_factor != 0:
#         raise ValueError('ny must be a multiple of acc_factor')

#     unmix1d = torch.zeros_like(csm1d)
#     n_blocks = ny // acc_factor
#     for index in range(n_blocks):
        
#         A = csm1d[index::n_blocks, :].T  # [nc, n_alias]
#         if torch.max(torch.abs(A)) > 0:
#             AHA = A.conj().T @ noise_matrix_inv @ A  # [n_alias, n_alias]
#             diag_mask = torch.abs(torch.diag(AHA)) > 0
#             reduced_eye = torch.diag(diag_mask.to(AHA.dtype))
#             n_alias = reduced_eye.sum().item()
#             scaled_reg_factor = regularization_factor * torch.trace(AHA)/ n_alias 
#             inv_term = torch.linalg.pinv(AHA + reduced_eye * scaled_reg_factor)
#             unmix_block = inv_term @ A.conj().T @ noise_matrix_inv  # [n_alias, nc]
#             unmix1d[index::n_blocks, :] = unmix_block
#     return unmix1d

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
    nx_blocks = nx // acc_factor_x  ##√
    ny_blocks = ny // acc_factor_y  ##√
   
    for ix_blocks in range(nx_blocks):
        for iy_blocks in range(ny_blocks):
            # Get aliased pixels for this block
            sub_csm = csm2d[ix_blocks::nx_blocks, iy_blocks::ny_blocks, :]##√
            A = sub_csm.reshape(acc_factor_x*acc_factor_y,-1)##√
            A = A.T##√
            if torch.max(torch.abs(A)) > 0.01:
                pixels_unaliased = torch.zeros(A.shape[1], A.shape[0], dtype=A.dtype, device=A.device)
                nonzeros_index = (torch.sum(torch.abs(A), dim=0) > 0.01)##√
                
                A_reduced = A[:, nonzeros_index]  ###√
                
                AHA = A_reduced.conj().T @ A_reduced  # [nc, nc]#####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!这个数是不对的哦
                
                diag_mask = torch.abs(torch.diag(AHA)) > 0.01
                reduced_eye = torch.diag(diag_mask.to(AHA.dtype))##√
                
                n_alias = reduced_eye.sum().item()
                scaled_reg_factor = regularization_factor * torch.trace(AHA) / n_alias 
               
                X = torch.linalg.solve(AHA + reduced_eye * scaled_reg_factor,A_reduced.conj().T)
                X_reshaped = X.reshape(A_reduced.shape[1], nc)
                pixels_unaliased[nonzeros_index, :] = X_reshaped#############√
                
                # Place result back
                
                block = pixels_unaliased.reshape(acc_factor_x, acc_factor_y, nc)
                unmix2d[ix_blocks::nx_blocks, iy_blocks::ny_blocks, :] = block
               


    return unmix2d