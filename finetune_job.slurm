#!/bin/bash -l
#SBATCH --job-name=donut-finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --partition=course_gpu
#SBATCH --gres=gpu:a100_10g:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tlw25@njit.edu
mkdir -p logs
module load bright
module load python3
source venv/bin/activate
python src/finetune_donut.py --dataset ./dataset --epochs 4 --out ./donut-bankstatement --batch-size 1
