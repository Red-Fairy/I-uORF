DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/room_diverse_nobg/train-4obj'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 5000 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'room_diverse_mask' \
    --display_port $PORT --display_ncols 4 --print_freq 100 --display_freq 100 --display_grad \
    --load_size 128 --n_samp 96 --input_size 128 --supervision_size 64 --frustum_size 64 \
    --model 'uorf_nogan_triplane_mask' \
    --num_slots 5 --attn_iter 3 --n_layer 2 \
    --shape_dim 48 --color_dim 16 \
    --bottom \
    --sam_encoder --encoder_size 1024 \
    --project \
    --coarse_epoch 25 --niter 50 --percept_in 10 \
    --attn_decay_steps 100000 \
    --exp_id '0503-mask-triplane-4obj' \
    --save_epoch_freq 10 \
    --dummy_info 'mask fg' --is_train \