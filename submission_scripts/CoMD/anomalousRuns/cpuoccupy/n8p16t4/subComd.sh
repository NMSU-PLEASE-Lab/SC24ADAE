#!/bin/bash


#SBATCH --job-name comd ## name that will show up in the queue
#SBATCH --output comd-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --nodes=8       # Total # of nodes
#SBATCH --time=00:30:00   # Total run time limit (hh:mm:ss)
#SBATCH -e comd-%j.error     # Name of stderr error file
#SBATCH -p wholenode     # Queue (partition) name
#SBATCH --exclusive

echo "anomalous run"

## Load modules
module load gcc/11.2.0
module load openmpi/4.0.6

export OMP_NUM_THREADS=4
export OMP_PLACES=cores 
export OMP_PROC_BIND=spread

EXECHPAS=/anvil/projects/x-cis230165/tools/HPAS/install/bin/hpas
EXEC=../../CoMD-openmp-mpi.appekg

srun --overlap --ntasks-per-node=64 --cpu-bind=map_cpu:0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126 ${EXECHPAS}  cpuoccupy -u 90 --verbose &
time srun --overlap --ntasks-per-node=16 --cpus-per-task=8 --cpu-bind=map_cpu:0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,40,42,44,46,48,50,52,54,56,58,60,62,64,66,68,70,72,74,76,78,80,82,84,86,88,90,92,94,96,98,100,102,104,106,108,110,112,114,116,118,120,122,124,126 ${EXEC} -e -i 8 -j 4 -k 4 -x 600 -y 600 -z 300


