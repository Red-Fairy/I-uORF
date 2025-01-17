#!/bin/bash
#SBATCH --account=viscam --partition=viscam,viscam-interactive,svl,svl-interactive --qos=normal
#SBATCH --nodes=1
##SBATCH --cpus-per-task=16
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

DATAROOT=${1:-'/svl/u/redfairy/datasets/real/kitchen-easy/4obj-all-test-0816'}
PORT=${2:-12783}
CUDA_VISIBLE_DEVICES=0 python test-slot-video.py --dataroot $DATAROOT --n_scenes 95 --n_img_each_scene 1 \
    --checkpoints_dir 'checkpoints' --name 'room_real_pots' --results_dir 'results' \
    --display_port $PORT --display_ncols 4 \
    --load_size 128 --input_size 64 --render_size 8 --frustum_size 128 \
    --n_samp 256 --z_dim 96 --num_slots 5 --near 6 --far 20 --bottom \
    --model 'uorf_eval' \
    --fixed_locality \
    --pos_emb --exp_id '/viscam/projects/uorf-extension/I-uORF/checkpoints/room_real_pots/ablation/uORF-4obj-GAN-fixed-real' \
    --attn_iter 3 --testset_name 'test_video_slot' --video --recon_only --no_loss \
# done
echo "Done"