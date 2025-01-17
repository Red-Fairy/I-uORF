#!/bin/bash

#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:a5000:1

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
DATAROOT=${1:-'/svl/u/redfairy/datasets/real/dataset-0721/1obj/dataset_undistorted_multiview'}
PORT=${2:-12783}
python -m visdom.server -p $PORT &>/dev/null &
python train_without_gan.py --dataroot $DATAROOT --n_scenes 51 --n_img_each_scene 2 \
    --checkpoints_dir 'checkpoints' --name 'room_real_pots' \
    --display_port $PORT --display_ncols 4 --print_freq 51 --display_freq 51 --save_epoch_freq 30 \
    --load_size 128 --n_samp 64 --input_size 128 --supervision_size 128 --frustum_size 128 \
    --model 'uorf_general' \
    --attn_decay_steps 200000 \
    --bottom \
    --encoder_size 896 --encoder_type 'DINO' \
    --num_slots 2 --attn_iter 4 --shape_dim 48 --color_dim 48 --near 6 --far 20 \
    --coarse_epoch 3000 --niter 3000 --percept_in 500 --no_locality_epoch 1000 --seed 2023 \
    --load_pretrain --load_pretrain_path '/viscam/projects/uorf-extension/I-uORF/checkpoints/room_real_pots/0724-new/1obj-scratch-pos-4848' \
    --load_encoder 'load_train' --load_slotattention 'load_train' --load_decoder 'load_train' \
    --exp_id '0721-real/1obj-loadScratch-ttt-CIT-dropout' \
    --position_loss --position_in 1000 \
    --fixed_locality --load_intrinsics \
    --color_in_attn --feat_dropout --feat_dropout_start 2000 \
    --obj_scale 4.5 --world_obj_scale 4.5 \
    --continue_train --epoch 2400 --epoch_count 2401 \
    --dummy_info 'DINO from 1 obj (scratch) with white BG. dim=48+48' \
    

# can try the following to list out which GPU you have access to
#srun /usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery

# done
echo "Done"
