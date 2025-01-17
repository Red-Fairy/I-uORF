DATAROOT=${1:-'/viscam/u/redfairy/room_diverse_generation/image_generation/datasets/1200shape_nobg-5000'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_diverse' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 50 --display_grad \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 \
    --model 'uorf_nogan_T_sam' \
    --num_slots 5 --attn_iter 4 \
    --z_dim 56 --texture_dim 8 \
    --bottom \
    --sam_encoder --encoder_size 1024 \
    --project \
    --lr 3e-4 --coarse_epoch 80  --niter 160 --percept_in 24 \
    --exp_id '0415-sam-texture-diverse-48-8' \
    --save_epoch_freq 10 \
    --seed 2023 \
    --dummy_info 'sam encoder v0' \