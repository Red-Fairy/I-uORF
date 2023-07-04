#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G

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
DATAROOT=${1:-'/viscam/redfairy/image_generation/datasets/ABO-chairs-4obj'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
CUDA_VISIBLE_DEVICES=0 python train_without_gan.py --dataroot $DATAROOT --n_scenes 3000 --n_img_each_scene 3 \
    --checkpoints_dir 'checkpoints' --name 'real_chairs' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 20 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 --frustum_size 64 \
    --model 'uorf_general' \
    --attn_decay_steps 200000 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 5 --attn_iter 4 --shape_dim 72 --color_dim 24 --freezeInit_steps 100000 \
    --coarse_epoch 100 --niter 200 --percept_in 50 --no_locality_epoch 50 --seed 2023 \
    --load_pretrain --load_pretrain_path '/viscam/redfairy/I-uORF/checkpoints/real_chairs/0628/1obj-scratch-1296-72-24' \
    --load_encoder 'load_train' --load_slotattention 'load_freeze' --load_decoder 'load_freeze' \
    --exp_id '0628/4obj-load-ttf-72-24-r2' \
    --dummy_info 'DINO, load from 1 obj with learnable position, ~200epoch, freezeInit 100000 (33epoch), percept & no-locality 50 epoch' \


# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"
