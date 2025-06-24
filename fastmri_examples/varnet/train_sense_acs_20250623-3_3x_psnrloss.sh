CUDA_VISIBLE_DEVICES=2 python train_SENSE_demo_Acs2SensMoedlv2_psnrloss.py \
    --racc 3 \
    --num_workers 0 \
    --challenge multicoil \
    --data_path 'COIL_FASTMRI_noshift_50' \
    --mask_type 'equispaced_fraction'