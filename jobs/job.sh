#!/bin/bash

#SBATCH --mail-user=ericdavid.boittier@unibas.ch
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=3000
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --nodelist=gpu20

hostname

#module load gcc/gcc4.8.5-openmpi1.10-cuda9.2

source ~/.bashrc

conda init bash

conda activate jaxe3xcuda11p39

which python
echo "GPU ID:" $CUDA_VISIBLE_DEVICES
python /pchem-data/meuwly/boittier/home/jaxeq/dcmnet/main.py --random_seed $RANDOM --n_dcm 2 --n_gpu $CUDA_VISIBLE_DEVICES 

#~/psi4conda/envs/jaxe3xcuda11py39/bin/python 

