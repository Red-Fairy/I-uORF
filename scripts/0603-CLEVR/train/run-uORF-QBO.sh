#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:titanrtx:1

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
DATAROOT=${1:-'/viscam/projects/uorf-extension/datasets/clevr_bg/train-4obj'}
PORT=${2:-8077}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1200 --n_img_each_scene 4  \
    --checkpoints_dir 'checkpoints' --name 'clevr_bg' \
    --display_port $PORT --display_ncols 4 --print_freq 50 \
    --load_size 128 --n_samp 64 --input_size 64 --supervision_size 64 \
    --coarse_epoch 600  --niter 1200 --percept_in 30 --no_locality_epoch 60 \
    --z_dim 32 --num_slots 8 --attn_iter 3 \
    --exp_id 'uORF-QBO' --learnable_slot_init \
    --model 'uorf_nogan' \
    --lr '3e-4' \
# done
echo "Done"