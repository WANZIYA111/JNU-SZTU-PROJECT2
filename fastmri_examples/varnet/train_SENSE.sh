CUDA_VISIBLE_DEVICES=0 python train_SENSE_demo_Acs2SensMoedlv2.py \
    --racc 3 \
    --num_workers 1 \
    --challenge multicoil \
    --data_path 'COIL_FASTMRI_noshift_50' \
    --mask_type 'equispaced_fraction'