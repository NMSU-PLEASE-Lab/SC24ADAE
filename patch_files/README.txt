# To compile/build the applications, do the following:
Download applications from their websites:

LAMMPS: https://github.com/lammps/lammps/archive/refs/tags/stable_2Aug2023.zip
miniAMR: https://github.com/Mantevo/miniAMR.git and git checkout 5758040bd430a93348f000e25021c993bbe368a4
miniFE: https://github.com/Mantevo/miniFE/archive/refs/tags/2.2.0.zip
Gadget2: https://wwwmpa.mpa-garching.mpg.de/gadget/gadget-2.0.7.tar.gz
CoMd: https://github.com/exmatex/CoMD/archive/v1.1.tar.gz
PENNANT: https://github.com/lanl/PENNANT/archive/refs/heads/master.zip 

To apply the patch, run:

# This is a dry-run and will not make any changes
patch --dry-run -ruN -d lammps-stable_2Aug2023 < lammps.patch

# if this works, then apply the patch.
patch -ruN -d lammps-stable_2Aug2023 < lammps.patch

