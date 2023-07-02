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
DATAROOT=${1:-'/viscam/redfairy/image_generation/datasets/ABO-chairs-1obj-transparent'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
CUDA_VISIBLE_DEVICES=2 python train_without_gan.py --dataroot $DATAROOT --n_scenes 1296 --n_img_each_scene 4 \
    --checkpoints_dir 'checkpoints' --name 'real_chairs' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 20 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 128 --frustum_size 128 \
    --model 'uorf_general_mask' \
    --attn_decay_steps 100000 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 1 --attn_iter 4 --shape_dim 48 --color_dim 16 --world_obj_scale 3 \
    --coarse_epoch 300 --niter 300 --percept_in 20 --no_locality_epoch 150 --centered \
    --exp_id '0628/1obj-mask' \
    --dummy_info 'DINO, 1 obj without BG' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"
