This code works for the Polaron-Polariton paper

## Submitting jobs
Create an empty directory, and go to the directory
To run the job, first copy everything in this folder to the directory you will run the job. Here <run_250129_Polaron>
then use the following command to rename the necessary files to the directory name

```
rename   run_250129_1DMultilayer    <your directory name>    run_250129_1DMultilayer*
```

in the "***_builder.py" file, change the following line

```
ScratchFolder = "/" + HOME + "/scratch/"
```

to the directory you want to save the jobs outputs (brun*'s)

Change the parameters in Model and "***_builder.py" files accordingly.

These parameters are changed in "***_builder.py"  
--In Model file:  
Nlayer  
NSteps  
E0  
dE  
NTraj  
γ  
--in Method file  
jobnum  


<span style="text-decoration:underline;">Note that you shouldn't change the parameter after getting data, for a new set of parameters, you have to open new folders. For example, if you are getting data for 1 trajectory per folder, then you shouldn't change it to another number of trajectories (The initial sampling randomness will be screwed up). Or if you get data for ω=1440/cm, to get data for the ω=720/cm, you have to create a new empty directory.</span>

In this ***_builder.py version, if you have an N job and each job needs 1 core, you can create a batch of ncore and submit those batches. For example, if you have N=990 and jobs, each using 1 core per job, then allocating 25 jobs to each batch you will have 39 batches of 25 jobs and 1 batch of 15 jobs.  


Using the following lines in the ***_builder.py file, you can update job configurations
#Configuration
partition_info=['share', 25, 256]; time_str='0-01:00:00' # = [partition name, ncores per virtual node, Memory in GB]
cores_per_node = partition_info[1]  # Cores per node
nb = 1  # Number of nodes per batch
c_perjob = 1  # Cores per job
mem = 3  # Memory per job in GB

 **I recommend not to change this parameter as they are set to best perform for FASTER, for other clusters (STAMPEDE, Lonsestar6 and EXPANCE) I will update new sbatch.tempelate files, as their launchers are different from FASTER**


 If you want to see the initial state before submitting jobs, I added this line in "***_builder.py" file 

```
#print("skipping batching!!!!!"); exit()
```

 Which you can uncomment, so the jobs will not be submitted and you can make sure the initial states are correct.


 After everything is ready (parameters, initial states, etc) use the following command to submit the jobs

```
sbatch ***_builder.py
```

 you can also use (python ***_buildr.py), but it is recommended not to use the login node.

 The way this ***_builder.py works is as follows,
 1. First create the "*.qjob" files
 2. add them all to a single file called .task (each line is bash "*.qjob")
 3. from the task read the lines group them in batches and create separate 
 ***_batch_###.sbatch 
 ***_batch_###.tasks
for each batch. Then it will send them to the launcher using 

```
sbatch  ***_batch_###.sbatch 
```

After all the jobs are DONE, you can use the following command to remove extra files in the builder file (DO DELETE RUN UNTIL ALL JOBS ARE DONE)

```
bash  cleanbuilder.sbatch
```

## Reading data 
1. Copy the "average_obs.py" to the folder outputs are save (directory of brun*s)
2. change the <PARAMETERS> to be same as ***_builder.py file
3. the output is the group velocity in the same directory as "average_obs.py", and the averaged density matrix in the "parnum*" folders.

## Some hints.
1. In this code number of bruns are controlled by "Rsample_list", by using the following line

```
if γ != 0 and R%30 != 0:
     vnum += 1
     continue  
```

THis will creat Rsample_list[i]/30 number of folder
2. Number o tranjectory per folder is controlled by

```
NTraj_list = np.array([0*np.heaviside(_, 0) + 1 for _ in γ_list], dtype=int)
```

DO NOT CHANGE THIS NUMBER RANDOMLY (NTraj_list), OTHERWISE YOU WILL RUIN ALL DATA.