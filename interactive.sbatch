#!/bin/bash
#
#SBATCH --job-name sinteractive
#SBATCH --partition rtx6000
#SBATCH --qos normal
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --gpus=0
#SBATCH --mem=24 --gpu 1G
#SBATCH --nodes=1
#SBATCH -o /home/ashtomer/projects/robustness/interactive_%J.out

#export NODE_TLS_REJECT_UNAUTHORIZED='0'

#
echo Fri 15 Aug 2025 13:36:46 IDT
launch='sleep 600'
echo " STARTING JOB WITH THE COMMAND:  $launch "
#module load cuda/11.1
while true; do
        eval $launch
done
