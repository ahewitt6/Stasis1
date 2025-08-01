#!/bin/bash

# the very first line is a special directive for Bash shell, do not remove
# lines that start with "#SBATCH" are special directives for Slurm
# other lines that start with "#" are comments

#SBATCH --account=ahewitt1
#SBATCH --job-name=grid_small  ## job name
#SBATCH -p free              ## use free partition
#SBATCH --nodes=1            ## use 1 node, don't ask for multiple
#SBATCH --ntasks=1           ## ask for 1 CPU
#SBATCH --mem-per-cpu=36G     ## ask for 1Gb memory per CPU
#SBATCH --error=/pub/ahewitt1/Stasis/Stasis_prym/me_small.job.err    ## Slurm error  file, %x - job name, %A job id
#SBATCH --out=/pub/ahewitt1/Stasis/Stasis_prym/me_small.job.out      ## Slurm output file, %x - job name, %A job id

# Run command hostname and assign output to a variable
hn=`hostname`
echo "Running job on host $hn"

# load python module
module load python
source /pub/ahewitt1/Stasis/.venv/bin/activate
# execute python script
#python FORESEE/Models/HNL/HNL_min_mid_max.ipynb
#jupyter nbconvert --execute --to notebook FORESEE/Models/HNL/HNL_Kaon_D_analysis.ipynb
#/pub/ahewitt1/Git_felix_new/FORESEE/Models/HNL/
#python /pub/ahewitt1/Git_felix_new/FORESEE/Models/HNL/HNL_Kaon_D_analysis.ipynb
#jupyter nbconvert --execute --to notebook FORESEE/Models/HNL/output_notebook.ipynb
#papermill /pub/ahewitt1/Git_felix_new/FORESEE/Models/HNL/HNL_Kaon_D_analysis.ipynb -p benchmark_ind 0 -p gen_ind 0
#for running a jupyter notebook
#for i in {0..20}
#do
#        # Run your Python script through .ipynb
#        #jupyter nbconvert --execute --to notebook FORESEE/Models/HNL/HNL_spin_corr_tau_atlas.ipynb
#3        jupyter nbconvert --execute --to notebook /pub/ahewitt1/Git_Felix_new_new/Alec/important/sort_later/prompt_atlas_analysis/HNL_neutrino_events_8_20_24.ipynb
#done

#for i in {0..20}
#do
#        papermill /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/test.ipynb /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/output_notebook_${i}.ipynb -p ind $i
#done

#a good test and it works
#for i in {0..20}
#do
#        papermill /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/test.ipynb /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/output_notebook_ind.ipynb -p ind $i --kernel python3
#done

#for i in {0..4}
#do
#        #papermill /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/gen_faser_atlas_events4.ipynb /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/output_notebook_ind_maj.ipynb -p ind $i -p majorana $True --kernel python3
papermill /pub/ahewitt1/Stasis/Stasis_prym/Alec_gradient_descent0.ipynb /pub/ahewitt1/Stasis/Stasis_prym/output_notebook_grad_descent_19.ipynb
#        #--kernel python3
#done

#for i in {0..81}
#do
#        papermill /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/gen_faser_atlas_events4.ipynb /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/output_notebook_ind_dir.ipynb -p ind $i -p majorana False
#        #--kernel python3
#done

#jupyter nbconvert --execute --to notebook /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/gen_faser_atlas_events24_chisq_con.ipynb

#jupyter nbconvert --execute --to notebook --output /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/outputnotebook.ipynb /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/gen_faser_atlas_events24_chisq_con.ipynb
#papermill /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/gen_faser_atlas_events32_prof.ipynb /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/outputnotebook.ipynb -p dtype "mm" 
#papermill /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/gen_faser_atlas_events32_chisq_asimov_atlas.ipynb /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/outputnotebook.ipynb -p dtype "md" 

#papermill /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/gen_faser_atlas_events24_chisq_con.ipynb /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/outputnotebook.ipynb -p dtype "dm" 
#papermill /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/gen_faser_atlas_events24_chisq_con_avg.ipynb /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/outputnotebook.ipynb -p dtype "dd" 
#papermill /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/gen_faser_atlas_events24_chisq_con_avg.ipynb /pub/ahewitt1/Git_Felix_9_24_24/FORESEE-main/Models/HNL/Alec/outputnotebook1.ipynb -p dtype "md" 

#for running a python script
#for i in {0..100}
#do
#        # Run your Python script with parameters i and j; here it corresponds to benchmark_ind and gen_ind
#        python FORESEE/Models/HNL/HNL_spin_corr_tau_atlas.py $i
#done

#python FORESEE/Models/HNL/HNL_Kaon_D_analysis.py 1 0
#python FORESEE/Models/HNL/Generate_all_LLP_spectra.py 0 0
###########for HNL_Kaon_D_analysis########
#for i in {2..2}
#do
#    for j in {0..0}
#    do
#            # Run your Python script with parameters i and j; here it corresponds to benchmark_ind and gen_ind
#            python FORESEE/Models/HNL/Generate_all_LLP_spectra.py $i $j
#    done
#done
################################