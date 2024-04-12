#!/bin/bash


#SBATCH --job-name miniAMR ## name that will show up in the queue
#SBATCH --nodes=4       # Total # of nodes
#SBATCH --time=00:30:00   # Total run time limit (hh:mm:ss)
#SBATCH -o miniAMR-%j.out    # Name of stdout output file
#SBATCH -e miniAMR-%j.error     # Name of stderr error file
#SBATCH -p wholenode     # Queue (partition) name
#SBATCH --exclusive

echo "anomalous run"

# Load Modules
module load gcc/11.2.0
module load openmpi/4.0.6

export OMP_NUM_THREADS=4
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

EXEC=../../../miniAMRomp.x.appekg
EXECHPAS=/anvil/projects/x-cis230165/tools/HPAS/install/bin/hpas

srun --overlap  --ntasks-per-node=32 --cpu-bind=map_cpu:0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124 ${EXECHPAS}  cpuoccupy -u 30 --verbose &
time srun --overlap --ntasks-per-node=8 -c 16 --cpu-bind=cores ${EXEC} --num_refine 4 --max_blocks 6000 --init_x 2 --init_y 2 --init_z 2 --npx 4 --npy 4 --npz 2 --nx 8 --ny 8 --nz 8 --num_objects 1 --object 2 0 -0.01 -0.01 -0.01 0.0 0.0 0.0 0.0 0.0 0.0 0.0009 0.0009 0.0009 --num_tsteps 800 --comm_vars 2


