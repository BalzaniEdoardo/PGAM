#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-14:30:00
#SBATCH --mem=20GB
#SBATCH --job-name=TL_pred_from_TD
#SBATCH --mail-type=END
#SBATCH --mail-user=jpn5@nyu.edu
#SBATCH --output=slurm_%j.out
#NAME=“GAM_fit”
#WORKDIR="${SCRATCH}/${NAME}"
#echo ${WORKDIR} ${NAME} ${IID}.job
module purge
#. /etc/profile.d/modules.sh
module load anaconda3/2020.02
module load r/intel/4.0.3
module load matlab/2021b
#source activate /home/jpn5/.local/lib/pycuda3.6
source /scratch/eb162/venv/bin/activate

PROGDIR=$SCRATCH/GAM_Repo/JP

#Check if running as an array job

if [[ ! -z "$SLURM_ARRAY_TASK_ID" ]]; then
        IID=${SLURM_ARRAY_TASK_ID}
fi

# Run the program

#echo ${WORKDIR} ${NAME} ${IID}.job
python $PROGDIR/fit_dataset_p10000.py $IID
                                                      
