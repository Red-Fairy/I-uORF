DATAROOT=${1:-'/viscam/u/redfairy/datasets/clevr567/train'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1000 --n_img_each_scene 3  \
    --checkpoints_dir 'checkpoints' --name 'clevr_567' \
    --display_port $PORT --display_ncols 4 --print_freq 200 --display_freq 50 --display_grad \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 --num_slots 8 \
    --coarse_epoch 600 --niter 1200 \
    --model 'uorf_nogan_T_sam' \
    --exp_id 'text-sam-texture' \
    --z_dim 32 --texture_dim 8 \
    --sam_encoder --encoder_size 1024 \
    --attn_iter 4 \
    --project \
    --dummy_info 'frozen sam encoder, share grid embed projection, correct deduct operation (before azi transform), move deduction after locality, add decoder MLP to z-slots projection (w/ residual), 4 round attn, use ImageNet ResNet18 encoder, pyramid upsample' \