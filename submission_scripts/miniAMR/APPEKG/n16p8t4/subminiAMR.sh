#!/bin/bash


#SBATCH --job-name miniAMR ## name that will show up in the queue
#SBATCH --nodes=16       # Total # of nodes
#SBATCH --time=00:30:00   # Total run time limit (hh:mm:ss)
#SBATCH -o miniAMR-%j.out    # Name of stdout output file
#SBATCH -e miniAMR-%j.error     # Name of stderr error file
#SBATCH -p wholenode     # Queue (partition) name
#SBATCH --exclusive
echo "good run"
# Load Modules
module load gcc/11.2.0
module load openmpi/4.0.6

export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

EXEC=../../miniAMRomp.x.appekg


#--cpus-per-task=4 --cpu_bind=cores
time srun --ntasks-per-node=8 -c 16 --cpu_bind=cores ${EXEC} --num_refine 4 --max_blocks 6000 --init_x 2 --init_y 2 --init_z 2 --npx 8 --npy 4 --npz 4 --nx 8 --ny 8 --nz 8 --num_objects 1 --object 2 0 -0.01 -0.01 -0.01 0.0 0.0 0.0 0.0 0.0 0.0 0.0009 0.0009 0.0009 --num_tsteps 800 --comm_vars 2


