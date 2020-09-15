#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=14:30:00
#SBATCH --mem=28GB
#SBATCH --job-name=TL_pred_from_TD
#SBATCH --mail-type=END
#SBATCH --mail-user=jpn5@nyu.edu
#SBATCH --output=slurm_%j.out
#NAME=“GAM_fit”
#WORKDIR="${SCRATCH}/${NAME}"
#echo ${WORKDIR} ${NAME} ${IID}.job
module purge
#. /etc/profile.d/modules.sh
#module load cudnn/8.0v6.0
#module load cuda/8.0.44

module load anaconda3/5.3.1
module load r/intel/3.6.0
#source activate /home/jpn5/.local/lib/pycuda3.6
source activate /home/jpn5/.local/lib/python3.7
PROGDIR=$SCRATCH/jp_final_gam_fit_coupling

#Check if running as an array job

if [[ ! -z "$SLURM_ARRAY_TASK_ID" ]]; then
        IID=${SLURM_ARRAY_TASK_ID}
fi

# Run the program

#echo ${WORKDIR} ${NAME} ${IID}.job

python $PROGDIR/gam_fit.py $IID template.npz
                                                      
