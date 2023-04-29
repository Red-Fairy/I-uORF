#!/bin/bash
DATAROOT=${1:-'/viscam/u/redfairy/datasets/clevr567/train'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'clevr_567' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 200 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --coarse_epoch 600 --z_dim 40 --num_slots 8 \
    --model 'uorf_nogan' \
    --exp_id 'uorf_original' \
    --dummy_info 'original implementation of uORF' \
# done
echo "Done"