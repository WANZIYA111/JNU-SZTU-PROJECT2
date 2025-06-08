conda activate py12 # this env contains python3.12, pytorch 2.7 and lastest lighting v2.x
python train_varnet_lightningv2.py --data_path "/data3/COIL_FASTMRI_noshift_50/" --num_worker 1 --devices 4
# if 16-mixed precision is needed, use the following command to use scale 10000, otherwise the data range is too small: 
python train_varnet_lightningv2.py --data_path "/data3/COIL_FASTMRI_noshift_50/" --num_worker 1 --devices 2 --scale 10000 --precision 16-mixed

# to test
python train_varnet_lightningv2.py   --mode test   --test_path /data3/COIL_FASTMRI_noshift_50/multicoil_test   --devices 1   --num_worker 4 --scale 10000 --ckpt_path varnet/lightning2/checkpoints/varnet-epoch\=116-val_loss\=0.0000.ckpt 