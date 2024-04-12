#!/bin/bash

#SBATCH --job-name Gadget-CLEAN ## name that will show up in the queue
#SBATCH --nodes=4       # Total # of nodes
#SBATCH --time=00:30:00   # Total run time limit (hh:mm:ss)
#SBATCH -o gadget-%j.out     # Name of stdout output file
#SBATCH -e gadget-%j.error     # Name of stderr error file
#SBATCH -p wholenode     # Queue (partition) name
#SBATCH --exclusive

module load gsl/2.4
#module load fftw/2.1.5 
module load gcc/11.2.0
module load openmpi/4.0.6

mkdir -p galaxy
mkdir -p parameterfiles
mkdir -p ICs

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/apps/spack/anvil/apps/fftw/2.1.5-gcc-8.4.1-cac36sv/lib
#export PeriodicBoundariesOn=1
cp /anvil/projects/x-cis230165/apps/Gadget-2.0.7/Gadget2/parameterfiles/galaxy.param .
cp /anvil/projects/x-cis230165/apps/Gadget-2.0.7/Gadget2/parameterfiles/outputs_lcdm_gas.txt parameterfiles
cp /anvil/projects/x-cis230165/testRuns/Gadget/ics-sizes/ic-s12.0 ICs
EXEC=../../Gadget2

time srun --ntasks-per-node=32 ${EXEC} galaxy.param
