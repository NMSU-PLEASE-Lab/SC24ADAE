#!/bin/bash


#SBATCH --job-name miniFEAPPEKG ## name that will show up in the queue
#SBATCH --nodes=2       # Total # of nodes
#SBATCH --time=00:30:00   # Total run time limit (hh:mm:ss)
#SBATCH -o minife-%j.out    # Name of stdout output file
#SBATCH -e minife-%j.error     # Name of stderr error file
#SBATCH -p wholenode     # Queue (partition) name
#SBATCH --exclusive
echo "good run"

# Load Modules
module load gcc/11.2.0
module load openmpi/4.0.6

export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

EXEC=../../miniFE.x.appekg

time srun --ntasks-per-node=16 --cpus-per-task=8 --cpu-bind=cores ${EXEC} -nx 1000 -ny 1000 -nz 1000
