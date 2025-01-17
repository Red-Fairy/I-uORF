#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:3090:1

#SBATCH --job-name="T_uORF"
#SBATCH --output=logs/%j.out

# only use the following if you want email notification
####SBATCH --mail-user=youremailaddress
####SBATCH --mail-type=ALL

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURMTMPDIR="$SLURMTMPDIR
echo "working directory = "$SLURM_SUBMIT_DIR

# sample process (list hostnames of the nodes you've requested)
DATAROOT=${1:-'/svl/u/redfairy/datasets/real/kitchen-hard-new/4obj-train-100'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 100 --n_img_each_scene 2 \
    --checkpoints_dir 'checkpoints' --name 'kitchen-hard' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 20 \
    --load_size 256 --n_samp 64 --input_size 128 --frustum_size_fine 256 \
    --supervision_size 64 --frustum_size 64 \
    --model 'uorf_general_IPE' \
    --attn_decay_steps 25000 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 5 --attn_iter 4 --shape_dim 72 --color_dim 24 --near 6 --far 20 \
    --coarse_epoch 750 --niter 3000 --percept_in 150 --no_locality_epoch 0 --seed 2027 \
    --load_pretrain --load_pretrain_path '/viscam/projects/uorf-extension/I-uORF/checkpoints/room_real_chairs/0824/4obj-load-IPE-nodense' \
    --load_epoch 80 \
    --load_encoder 'load_train' --load_slotattention 'load_train' --load_decoder 'load_train' \
    --fixed_locality --dense_sample_epoch 500 --depth_supervision --depth_in 2000 \
    --stratified --fg_object_size 3 --n_dense_samp 256 \
    --bg_density_loss \
    --continue_train --epoch_count 791 \
    --exp_id '0828/4obj-loadchairs-fine256-100scenes' \
    --dummy_info 'DINO load from 4 obj chairs synthetic' \
    
# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"
