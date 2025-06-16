import os
import h5py
import numpy as np
import mriutils
import torch
import fastmri
from mrfftv2 import ifftc,ifft2c,fftc,fft2c,sos
os.environ["BART_PATH"]    = 'bart'
os.environ["OMP_NUM_THREADS"] = "1"

folder_path = '/data3/wangyuwandata/COIL_FASTMRI_noshift_50/multicoil_train'

    
# 处理 kspace 的示例函数（替换为你的实际处理方法）
def process_kspace(new_kspace):
    tmp_sens_maps = []
    tmp_weight = []
    for slice_index in range(new_kspace.shape[0]):
        slice_kspace = new_kspace[slice_index]
        # coilsen = mriutils.bart(1, 'ecalib  -m1 -W -c0', slice_kspace.transpose((1, 2, 0))[None,...])[0].transpose(2,0,1)#(16, 384, 384)
        sens_map, weights = mriutils.bart(2, 'ecalib -m1 -W -c0', slice_kspace.transpose((1,2,0))[None,...])
        coilsen = sens_map[0].transpose(2,0,1)  # (16,384,384)  
        tmp_sens_maps.append(coilsen[None])
        tmp_weight.append(weights)
    gold_sens_maps = np.concatenate(tmp_sens_maps, axis=0) 
    gold_weights = np.concatenate(tmp_weight, axis=0)
    return gold_sens_maps,gold_weights
    
def center_crop(data, shape):
    if not (0 < shape[0] <= data.shape[-2] and 0 < shape[1] <= data.shape[-1]):
        raise ValueError("Invalid shapes.")

    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]

    return data[..., w_from:w_to, h_from:h_to]
# 遍历文件夹中的所有 h5 文件
for file_name in os.listdir(folder_path):
    numbers = 0
    if file_name.endswith(".h5"):
        file_path = os.path.join(folder_path, file_name)
        
        # 打开 h5 文件
        with h5py.File(file_path, 'r+') as h5_file:  # 使用 'r+' 以允许读写
            if 'kspace' in h5_file:
                kspace = h5_file['kspace'][()]  # 读取 kspace 数据
                reconrss_data = h5_file['reconstruction_rss'][()]
                raw_image = ifft2c(kspace)
                size = min(reconrss_data.shape[-2],reconrss_data.shape[-1])
                new_image = center_crop(raw_image,(size,size))
                new_kspace = fft2c(new_image)
                # 处理 kspace 并生成 sens_map
                sens_map,weights = process_kspace(new_kspace)
                
                if abs(sens_map).max()>1:
                    numbers = numbers+1
                # 如果 'sens_map' 已存在，先删除
                if 'sens_map' in h5_file:
                    del h5_file['sens_map']

                if 'weights' in h5_file:
                    del h5_file['weights']
                    
                # 存储处理后的数据
                h5_file.create_dataset('sens_map', data=sens_map)
                h5_file.create_dataset('weights', data=weights)
                
                print(f"处理并保存 {file_name} 完成！")

print("所有文件处理完成！🚀")
print("灵敏度图大于1的文件个数为：",numbers)