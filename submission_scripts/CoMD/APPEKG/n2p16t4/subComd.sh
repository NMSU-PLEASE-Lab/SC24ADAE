#!/bin/bash


#SBATCH --job-name comd ## name that will show up in the queue
#SBATCH --output comd-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --nodes=2       # Total # of nodes
#SBATCH --time=00:30:00   # Total run time limit (hh:mm:ss)
#SBATCH -e comd-%j.error     # Name of stderr error file
#SBATCH -p wholenode     # Queue (partition) name
#SBATCH --exclusive

echo "normal run"

## Load modules
module load gcc/11.2.0
module load openmpi/4.0.6

export OMP_NUM_THREADS=4
export OMP_PLACES=cores 
export OMP_PROC_BIND=spread

EXEC=../../CoMD-openmp-mpi.appekg

time srun --ntasks-per-node=16 --cpus-per-task=8 --cpu-bind=cores ${EXEC} -e -i 4 -j 4 -k 2 -x 300 -y 300 -z 300
