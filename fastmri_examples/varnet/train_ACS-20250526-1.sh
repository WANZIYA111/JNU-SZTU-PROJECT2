CUDA_VISIBLE_DEVICES=5 python train_varnet_demo_Acs2SensMoedl.py \
    --racc 7 \
    --challenge multicoil \
    --data_path 'COIL_FASTMRI_noshift_50' \
    --mask_type 'equispaced_fraction'