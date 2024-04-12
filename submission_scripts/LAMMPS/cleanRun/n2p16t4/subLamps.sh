#!/bin/bash


#SBATCH --job-name LAMMPSAPPEKG ## name that will show up in the queue
#SBATCH --nodes=2       # Total # of nodes
#SBATCH --time=00:30:00   # Total run time limit (hh:mm:ss)
#SBATCH -J lammps     # Job name
#SBATCH -o lammps-%j.out     # Name of stdout output file
#SBATCH -e lammps-%j.error     # Name of stderr error file
#SBATCH -p wholenode     # Queue (partition) name
#SBATCH --exclusive

## Load modules
module load gcc/11.2.0
module load openmpi/4.0.6


INPUT=in.ar.lj
EXEC=../../lmp_omp

export OMP_NUM_THREADS=4 #$SLURM_CPUS_PER_TASK
export OMP_PROC_BIND=spread
export OMP_PLACES=cores
time srun --ntasks-per-node=16 -c 8 --cpu-bind=cores ${EXEC} -sf omp -pk omp 4 -in ${INPUT}
