#!/bin/bash
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room-real/chairs/train-1obj'}
PORT=${2:-12783}
CUDA_VISIBLE_DEVICES=0 python test-video.py --dataroot $DATAROOT --n_scenes 1 --n_img_each_scene 1 \
    --checkpoints_dir 'checkpoints' --name 'room_diverse' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 128 --render_size 8 --frustum_size 128 \
    --n_samp 256 --num_slots 2 \
    --model 'uorf_general_eval' \
    --encoder_size 896 --encoder_type 'DINO' \
    --exp_id '/viscam/projects/uorf-extension/I-uORF/checkpoints/room_real_chairs/1obj-scratch-posLoss-nss' \
    --shape_dim 72 --color_dim 24 --bottom \
    --random_init_pos \
    --attn_iter 4 --no_loss --visual_only --visual_idx 0 --testset_name test_video-swapToOrigin-randomPos/idx0 \
# done
echo "Done"