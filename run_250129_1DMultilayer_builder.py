#!/sw/eb/sw/Anaconda3/2024.02-1/bin/python                                                                                         

##NECESSARY JOB SPECIFICATIONS                                                                                                     
#SBATCH --partition=shared                                                                                                         
#SBATCH --job-name=250127...builder         #Set the job name to "JobExample1"                                                
#SBATCH --time=0-04:00:00          #Set the wall clock limit to 1hr and 30min                                                      
#SBATCH --ntasks=1               #Request 1 task                                                                                   
#SBATCH --nodes=1                                                                                                                  
#SBATCH --cpus-per-task=1                                                                                                          
#SBATCH --mem=1G                                                                                                                   
#SBATCH --output=output/out.%j  

import subprocess
import concurrent.futures
import os
import numpy as np
import time

current_directory = os.getcwd()
HOME = os.path.join(*current_directory.split('/')[:4])#"/scratch/user/u.sr215595/"
project_name=os.getcwd().split('/')[-3]
JobName=os.getcwd().split('/')[-1]
LOCAL_DIR=os.getcwd()#HOME + "/Projects/" + project_name + "/build/" + JobName
ScratchFolder = "/" + HOME + "/scratch/" # Directory you save the data



logstr='''This job:
For L=18: if l == 18 and G<0.3 and vnum%4==0 and Ee<0:

***write me***

Codebase:
Cluster: FASTER
gittag: '''
open(JobName+'.log','w').write(logstr)

#Setup the versionmap and qjob files
vmap_file=open('versionmap.dat','w')
vmap_file.write('vnum\tL\n')

task_file=open(JobName+'.task','w')
template_file=JobName+'.template'
template_contents=open(template_file,'r').read()


##############################################
################# PARAMETERS #################
# a.u transfer of units
amu, ps, cm, Å =  1836.0, 41341.37, 1/0.000004556, 1.8897259885789
c, π = 137.036, np.pi
c = c/2.4
eV = 1/27.2114


modelfile = 'EP1D'
methodfile = 'splitop'#['mfesin', 'splitop']
initialfile = 'InitialMC'
a = 12 * Å
Nskip = 15
Nsite = 40001
dtN = 20


# Modifying Parameters
NSteps = 501
nl_list = [1]
γ_list = np.round(np.arange(0.0, 6.1, 0.5), 3)
E0dE_tresh = 3.0
E0_list = np.round(np.arange(2.36, 2.91, 0.01),3)
dE_list = np.round(np.heaviside(-E0_list + E0dE_tresh, 0)*0.04+0.01, 3) # half energy window
branch = 0 #"lower" or "upper"
NTraj_list = np.array([0*np.heaviside(_, 0) + 1 for _ in γ_list], dtype=int)
Rsample_list = np.array([299*np.heaviside(_, 0) + 1 for _ in γ_list], dtype=int) 
print("E0_list: ", E0_list)
print("dE_list: ", dE_list)
print("γ_list: ", γ_list)
print("NTraj_list: ", NTraj_list)
print("Rsample_list: ", Rsample_list)#
################# PARAMETERS #################
##############################################

############################################################
################# Generating initial state #################                                                                                                        
vnum=0
os.system("mkdir -p InitialStates")
os.system("mkdir -p InitialGenerators")
for E0, dE in zip(E0_list, dE_list):
        Initgenfile = f'InitGentor_E0={E0:.2f}_dE={int(dE * 1000)}.npy'
        if os.path.isfile(LOCAL_DIR + '/InitialStates/'+Initgenfile) == False:
            print("vnum = ", vnum)
            print(f'files ' + Initgenfile +  ' doesnt exist')
            os.chdir(LOCAL_DIR)
            Model_contents=open(f"./Model/{modelfile}.py",'r').read()
            os.chdir(LOCAL_DIR+'/InitialGenerators')
            os.system("mkdir -p  ./init"+str(vnum))
            os.chdir(LOCAL_DIR+f'/InitialGenerators/init{vnum}')
            # copy                                                                                                                 
            os.system("cp " + LOCAL_DIR + "/InitialMC.py ./")
            os.system("cp " + LOCAL_DIR + "/input.txt ./")
            os.system("mkdir -p Model")

            # Edit model file                                                                                                      
            fout=open(f"./Model/{modelfile}.py",'w')
            contents=Model_contents.replace('***Nlayer***', str(1))
            contents=contents.replace('***dtN***', str(dtN))
            contents=contents.replace('***Nsite***', str(Nsite))
            contents=contents.replace('***NSteps***', str(NSteps))
            contents=contents.replace('***Nskip***', str(Nskip))
            contents=contents.replace('***dE***', str(dE))
            contents=contents.replace('***E0***',str(E0))
            contents=contents.replace('***NTraj***', str(1))
            contents=contents.replace('***γ***', str(0))
            fout.write(contents)
            fout.close()

            os.system("sbatch InitialMC.py Findwfseed")
        vnum+=1                                                                               

firstcheck = 1
running_jobs = 1
while running_jobs > 0:
    print("Waiting for jobs to finish...")
    running_jobs = 0
    vnum = 0
    time.sleep(3)
    # submited_jobs = os.popen("squeue -u u.sr215595").read()                                                                      
    # submited_jobs = [int(_.split()[0]) for _ in submited_jobs.split('\n')[1:-1]]                                                 
    if firstcheck == 1:
        submited_jobs = []
    running_jobs = 0
    for E0, dE in zip(E0_list, dE_list):
        Initgenfile = f'InitGentor_E0={E0:.2f}_dE={int(dE * 1000)}.npy'
        # print(LOCAL_DIR + '/InitialStates/'+Initgenfile, os.path.isfile(LOCAL_DIR + '/InitialStates/'+Initgenfile) == True)
        # if os.path.isfile(LOCAL_DIR + '/InitialStates/'+Initgenfile) == True:
        #     continue
        if os.path.isdir(LOCAL_DIR + '/InitialGenerators/init'+str(vnum)) == True:
            if os.path.isfile(LOCAL_DIR + '/InitialGenerators/init'+str(vnum)+f'/InitialStates/{Initgenfile}') == False:
                submited_jobs.append(vnum)
                running_jobs += 1
            if os.path.isfile(LOCAL_DIR + '/InitialGenerators/init'+str(vnum)+f'/InitialStates/{Initgenfile}') == True:
                os.chdir(LOCAL_DIR+f'/InitialGenerators/init{vnum}')
                os.system('cp  ./InitialStates/InitGentor*  '+LOCAL_DIR+'/InitialStates')
                os.system('cp  ./InitialStates/Convergance_E0*  '+LOCAL_DIR+'/InitialStates')
                os.system('cp  ./InitialStates/Photonabswf_E0*  '+LOCAL_DIR+'/InitialStates')
                os.system('cp  ./InitialStates/Excitonabswf_E0*  '+LOCAL_DIR+'/InitialStates')
                os.chdir(LOCAL_DIR)
        vnum += 1
    print("Running jobs vnums = ", submited_jobs)


print("Initial states done!")
os.system("rm -r "+LOCAL_DIR+'/InitialGenerators')
#exit() 
################# Generating initial state #################
############################################################

#########################################################
################## Generate qjob files ##################
vnum=0
vnumt=0
for nl in nl_list:
    for γ, NTraj, Rsample in zip(γ_list, NTraj_list, Rsample_list):
        for E0, dE in zip(E0_list, dE_list):
            for R in range(Rsample):
                # if E0 not in [2.75, 2.76, 2.77]:
                #     vnum += 1
                #     continue
                if nl not in [1]:
                    vnum += 1
                    continue
                if γ not in [1.5]:
                    vnum += 1
                    continue
                if γ != 0 and R%30 != 0:
                    vnum += 1
                    continue  
                qjob_file=template_file.replace('.template','_'+str(vnum)+'.qjob')
                fout=open(qjob_file,'w')
                # *Project*   *JobName*   *modelfile*   *methodfile*   *Nlayer*   *NSteps*   *E0*  *dE*  *NTraj*  *γ*  *jobnum*
                contents=template_contents.replace('###',str(vnum))
                contents=contents.replace('*ScratchFolder*', ScratchFolder)
                contents=contents.replace('*Project*', project_name)
                contents=contents.replace('*JobName*', JobName)
                contents=contents.replace('*modelfile*', modelfile)
                contents=contents.replace('*methodfile*', methodfile)
                contents=contents.replace('*Nlayer*', str(nl))
                contents=contents.replace('*NSteps*', str(NSteps))
                contents=contents.replace('*E0*', f'{E0:.2f}')
                contents=contents.replace('dE=*dE*', "dE="+str(int(dE * 1000)))
                contents=contents.replace('*dE*', str(dE))
                contents=contents.replace('*NTraj*', str(NTraj))
                contents=contents.replace('*γ*', str(γ))
                contents=contents.replace('*jobnum*',str(str(int(vnum * NTraj * 1.2))))
                vmap_file.write(str(vnum)+'\t'+str(nl)+'\t'+str(γ)+'\t'+str(E0)+'\n')
                #print('bash '+JobName+'_'+str(vnum)+'.qjob')
                task_file.write('bash '+JobName+ '_'+str(vnum)+'.qjob\n')
                vnumt+=1
                fout.write(contents)
                fout.close()
                vnum+=1

print("-vnum", vnum)
print("-vnumt", vnumt)
print("__________________qjob files created__________________")

# Finally output sbatch file
contents=open(JobName + '.sbatch.template','r').read()
contents=contents.replace('*JobDetails*',str('n2 = 50 and n3 = 1600 '))
open(JobName+'.sbatch','w').write(contents)
################## Generate qjob files ##################
#########################################################

#print("skipping batching!!!!!"); exit()

############################################################
################### Submiting in batches ###################
# Configuration
partition_info=['share', 25, 256]; time_str='0-01:00:00' # = [partition name, ncores per virtual node, Memory in GB]
JobName = os.getcwd().split('/')[-1]
task_file = JobName + '.task'
sbatch_template = JobName + '.sbatch'
cores_per_node = partition_info[1]  # Cores per node
nb = 1  # Number of nodes per batch
c_perjob = 1  # Cores per job
mem = 3  # Memory per job in GB
if mem * partition_info[1]/c_perjob > partition_info[2]:
    print(f"\'Error in MEMORY: Memory per job is too high! {mem * partition_info[1]/c_perjob} > {partition_info[2]}\'")
    exit()

# Calculate the number of jobs per batch
jobs_per_batch = cores_per_node * nb // c_perjob

# Read the task file
with open(task_file, 'r') as file:
    tasks = file.readlines()

# Calculate the total number of jobs
total_jobs = len(tasks)

# Loop through the tasks and create batches
for i in range(0, total_jobs, jobs_per_batch):
    batch_tasks = tasks[i:i + jobs_per_batch]
    batch_index = i // jobs_per_batch + 1

    # Adjust the number of nodes for the last batch if necessary
    if len(batch_tasks) < jobs_per_batch:
        nb = (len(batch_tasks) + cores_per_node - 1) // cores_per_node  # Calculate the required number of nodes

    # Create a new task file for the batch
    batch_task_file = f'{JobName}_batch_{batch_index}.task'
    with open(batch_task_file, 'w') as file:
        file.writelines(batch_tasks)

    # Modify the sbatch file for the batch
    batch_sbatch_file = f'{JobName}_batch_{batch_index}.sbatch'
    with open(sbatch_template, 'r') as file:
        sbatch_content = file.read()

    print("jobs_per_batch: ", len(batch_tasks))
    sbatch_content = sbatch_content.replace('*nnn*', str(len(batch_tasks)))
    sbatch_content = sbatch_content.replace('*NNN*', str(nb))
    sbatch_content = sbatch_content.replace('*ccc*', str(c_perjob))
    sbatch_content = sbatch_content.replace('*mmm*', str(mem) + 'G')
    sbatch_content = sbatch_content.replace('*ttt*', time_str)
    sbatch_content = sbatch_content.replace('*TaskFile*', batch_task_file)
    sbatch_content = sbatch_content.replace('*JobName*', JobName+'_batch_'+str(batch_index))
    sbatch_content = sbatch_content.replace('*partition*',partition_info[0])
    

    with open(batch_sbatch_file, 'w') as file:
        file.write(sbatch_content)

    # Submit the batch job
    os.system(f'sbatch {batch_sbatch_file}')

print(f'cores_per_node: {cores_per_node}')
print(f'Total number of tasks: {total_jobs}')
print(f'Jobs per batch: {jobs_per_batch}')
print(f'Number of batches: {batch_index}')

################### Submiting in batches ###################
############################################################