#!/bin/bash
DATAROOT=${1:-'/viscam/u/redfairy/datasets/clevr567/test'}
PORT=${2:-12783}
python test.py --dataroot $DATAROOT --n_scenes 500 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'clevr567' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --render_size 8 --frustum_size 128 \
    --n_samp 256 --num_slots 8 \
    --model 'uorf_eval_T' \
    --sam_encoder --encoder_size 1024 \
    --pos_emb --exp_id '0409-sam-v1' \
    --z_dim 40 \
    --project --attn_iter 4 --testset_name trail_test \
# done
echo "Done"