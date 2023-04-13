#!/bin/bash
DATAROOT=${1:-'/viscam/u/redfairy/datasets/room_chair/test'}
PORT=${2:-12783}
python test-swapTexture.py --dataroot $DATAROOT --n_scenes 1 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'room_chair' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 64 --render_size 8 --frustum_size 128 \
    --n_samp 256 --z_dim 64 --num_slots 5 \
    --model 'uorf_eval_T_sam' \
    --sam_encoder --encoder_size 1024 \
    --exp_id '0410-sam-texture' \
    --z_dim 48 --texture_dim 16 \
    --project --attn_iter 4 --testset_name swap_slot_test \
# done
echo "Done"