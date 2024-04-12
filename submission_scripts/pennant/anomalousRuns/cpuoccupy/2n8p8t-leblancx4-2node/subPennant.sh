#!/bin/bash
#SBATCH --job-name PennantAPPEKG ## name that will show up in the queue
#SBATCH --output pennant-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --nodes=2       # Total # of nodes
#SBATCH --time=00:30:00   # Total run time limit (hh:mm:ss)
#SBATCH -p wholenode     # Queue (partition) name
#SBATCH --exclusive

## Load modules
module load gcc/11.2.0
module load openmpi/4.0.6

export OMP_NUM_THREADS=8
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

INPUT=../leblancx4-2node.pnt
EXEC=../pennant.APPEKG
EXECHPAS=/anvil/projects/x-cis230165/tools/HPAS/install/bin/hpas

echo "pennant.APPEKG CPUOCCUPY jobID $SLURM_JOB_ID"
pwd
echo -e "srun --overlap --ntasks-per-node 8 --cpu-bind=map_cpu:0,4,8,16,64,68,72,76 ${EXECHPAS} cpuoccupy -u 90 -d -1.0 -t 0.0 & \n"
echo -e "time srun --overlap --ntasks-per-node 8 -c 16 --cpu-bind=cores ${EXEC} ${INPUT} \n"

srun --overlap --ntasks-per-node 8 --cpu-bind=map_cpu:0,4,8,16,64,68,72,76 ${EXECHPAS} cpuoccupy -u 90 -d -1.0 -t 0.0 &
time srun --overlap --ntasks-per-node 8 -c 16 --cpu-bind=cores ${EXEC} ${INPUT}

echo -e "\nseff stats"
seff $SLURM_JOB_ID
