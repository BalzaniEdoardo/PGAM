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
module load anaconda3/2020.02
module load r/intel/4.0.3
#source activate /home/jpn5/.local/lib/pycuda3.6
source /scratch/jpn5/select_hand_vel/venv/bin/activate

PROGDIR=$SCRATCH/jp_final_gam_fit_coupling

#Check if running as an array job

if [[ ! -z "$SLURM_ARRAY_TASK_ID" ]]; then
        IID=${SLURM_ARRAY_TASK_ID}
fi

# Run the program

#echo ${WORKDIR} ${NAME} ${IID}.job

python $PROGDIR/gam_fit.py $IID template.npz
                                                      
