CUDA_VISIBLE_DEVICES=5 python train_varnet_demo_Acs2SensMoedlv2.py \
    --racc 7 \
    --num_workers 0 \
    --challenge multicoil \
    --data_path 'COIL_FASTMRI_noshift_50' \
    --mask_type 'equispaced_fraction'