#!/bin/bash
#SBATCH -t 3-00:00:00
#SBATCH -e outputs/f-mixpool51-skip-error.out
#SBATCH -o outputs/f-mixpool51-skip-output.out
#SBATCH --mem=80G
#SBATCH -J Train-skip
#SBATCH -c 16
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gpus=1
OPTIONS=' --image-path=/cluster/projects/radiomics/Temp/RADCURE-npy/img/       \
        --labels-path=/cluster/home/dzhu/RADCURE-classifier/data/labels.csv  \
        --weighted-sampling --input-scale-size=256 \
        --save-path=/cluster/home/dzhu/RADCURE-classifier/CNN/checkpoints/CNN-F/ \
        --train --epochs=15 \
        --linear-layers=5 --conv-layers=6 --skip-layers \
        --dropout-rate=0.0 --leakyrelu-param=0.05 \
        --learning-rate=0.09 --momentum=0.9 --weight-decay=0.0001 \
        --step-size=1 --lr-gamma=0.75 \
        --test --test-model-epoch=15 \
        --debug=1 --print-period=40 \
        --train-ratio=0.5 --train-batch-size=8 --train-workers=4 \
        --test-batch-size=4 --test-workers=4'
echo 'Starting Shell Script'
source /cluster/home/dzhu/.bashrc
python main.py $OPTIONS
echo 'Python script finished.'
