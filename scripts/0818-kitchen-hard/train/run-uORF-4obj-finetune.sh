#!/bin/bash

#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

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

DATAROOT=${1:-'/svl/u/redfairy/datasets/real/kitchen-hard-new/4obj-all-train'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 1 --start_scene_idx 20 --n_img_each_scene 1 \
    --checkpoints_dir 'checkpoints' --name 'kitchen-hard' \
    --display_port $PORT --display_ncols 4 --print_freq 50 --display_freq 50 --save_epoch_freq 250 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 64 \
    --load_pretrain --load_pretrain_path '/viscam/projects/uorf-extension/I-uORF/checkpoints/kitchen-easy/uORF-4obj-intrinsics' \
    --load_epoch 1140 --load_encoder 'load_train' --load_slotattention 'load_train' --load_decoder 'load_train' \
    --coarse_epoch 0 --niter 10000 --no_locality_epoch 0 --z_dim 96 --num_slots 5 --near 6 --far 20 \
    --model 'uorf_nogan' --bottom --fixed_locality \
    --exp_id 'uORF-4obj-intrinsics-zeroshot-20-oneview' \
# done
echo "Done"
