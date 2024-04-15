
# This script finds the quantitave quality values of lammps and minife.
# It computes two quality values:
#   -   quality over number of processes
#       quality = problemSize / (#processes * runTime)
#   -   quality over number of nodes
#       quality = probSize / (#nodes * runTime)
#       problemSize = (x * y * z) * #iterations
# to run the script, you should enter the file name. For example to compute the quantitave quality value of lammps
# you need to specify the submission script, slurm output file and the input file.
# For example, to run compute the quality value of a lammps job, run:
#   python3 /path/to/jobQuality.py -i /path/to/submissonScript /path/to/slurmOutput
# To use it for miniFE, don't include an input file. Also, use '-a' with 'minife'
#
# NOTE: This is an intial version. More apps maybe included.

import argparse 


# Parse arguments
parser = argparse.ArgumentParser(prog="minifeQuality.py", description='This script computes the quality matrics of jobs. Currentlly it only considers lammps and minife')

# Define how a single command-line argument should be parsed.
parser.add_argument('--application', '-a', type=str, required=False, help="Specify application name")
parser.add_argument('--input', '-i', type=str, required=True, nargs='+', help="Submission script and slurm output")
# parser.add_argument('--stats', '-s', type=str, required=True, choices=['descriptive','t-test', 'compare'], help="Select what statistics to generate. descriptive or t-test")

# extract the required from the minife submission script such as #nodes, #processes and input
# it computes  the problem size by first finding the input size and multiply it by the number of iterations(200) as follows:
#   size = nx * ny * ny
#   problemSize = size * 200
# it returns problem size, #nodes, and #allprocesses
def getMinifeResouces(script):
    f = open(script, "r")
    lines = f.readlines()
    for line in lines:
        if '-N' in line:
            words = line.split()
            nodes = int(words[2])
        if '--ntasks-per-node' in line and 'nx' in line and 'overlap' in line:
            words = line.split()
            procsPerNode = int(words[4])
            size = int(words[8]) * int(words[10]) * int(words[12])
            probSize = size * 200
            allProcs = procsPerNode * nodes
        if '--ntasks-per-node' in line and 'nx' in line and 'overlap' not in line:
            words = line.split()
            procsPerNode = int(words[3])
            size = int(words[6]) * int(words[8]) * int(words[10])
            probSize = size * 200
            allProcs = procsPerNode * nodes
        # if "srun" in line and 'nx' in line:
        #     words = line.split()
        #     size = int(words[6]) * int(words[8]) * int(words[10])
        #     probSize = size * 200
        #     allProcs = int(words[3]) * nodes
    f.close()
    return(probSize, nodes, allProcs)

def getLammpsResouces(script):
    f = open(script, "r")
    lines = f.readlines()
    for line in lines:
        if '-N' in line:
            words = line.split()
            nodes = int(words[2])
        if '--ntasks-per-node' in line and '-i' in line and 'overlap' in line:
            words = line.split()
            procsPerNode = int(words[4])
        if '--ntasks-per-node' in line and '-i' in line and '--overlap' not in line:
            words = line.split()
            procsPerNode = int(words[2])
    f.close()
    allProcs = nodes * procsPerNode
    return( nodes, allProcs)

# get problem parameters from the lammps input file
# it currentlly computes and returns the problem size:
#   probSize = (x*y*z) * runs
def getLammpsResources(inputFile):
    f2 = open(inputFile, 'r')
    lines = f2.readlines()
    for line in lines:
        if 'x index' in line:
            words = line.split('x index')
            x = int(words[1])
        if 'y index' in line:
            words = line.split('y index')
            y = int(words[1])
        if 'z index' in line:
            words = line.split('z index')
            z = int(words[1])
        if 'run' in line:
            words = line.split()
            runs = int(words[1])
    inputSize = x * y * z
    probSize = inputSize * runs
    return(probSize)


        

# get required info from the slurm output file
# currerntly it computes and returns the run time of the job in seconds
def getOutputInfo(outputFile):
    f1 = open(outputFile, "r")
    lines2 = f1.readlines()
    for line in lines2:
        if 'real' in line:
            words = line.split()
            # get minutes
            m = int(words[1].split('m')[0])
            # get seconds
            s = int(words[1].split('m')[1].split('.')[0])
            
        elif 'wall time' in line:
            words = line.split(':')
            m = int(words[2])
            s = int(words[3])
    runTime = (m*60) + s   
    f1.close()
    return(runTime)


args = parser.parse_args()
subScript = args.input[0] 
outFile = args.input[1]
app = args.application
if app == 'lammps' or app == None:
    inputFile = args.input[2]







if app == None or app == 'lammps':
    probSize = getLammpsResources(inputFile)
    wallTime = getOutputInfo(outFile)
    numNodes,numAllProcs = getLammpsResouces(subScript)
elif app == 'minife':
    probSize,numNodes,numAllProcs = getMinifeResouces(subScript)
    wallTime = getOutputInfo(outFile)
else:
    print("Currently we analyze lammps and minife only")




print("Number of all nodes = {}".format(numNodes))
print("Number of all processes = {}".format(numAllProcs))
print("problem size = {}".format(probSize))
print('Run time = {}s'.format(wallTime))
qualityProcs = probSize/(numAllProcs * wallTime)
print('Quality over all process = {}'.format(qualityProcs))
qualityNodes = probSize/(numNodes * wallTime)
print('Quality over all nodes = {}'.format(qualityNodes))

