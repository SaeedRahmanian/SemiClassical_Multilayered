#!/sw/eb/sw/Anaconda3/2024.02-1/bin/python

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=ObsAverag         #Set the job name to "JobExample1"
#SBATCH --time=0-10:00:00          #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1               #Request 1 task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5500M       


import numpy as np
import os
from scipy.stats import linregress


parnum = 0

modelfile = 'EP1D'
methodfile = 'splitop'#['mfesin', 'splitop']

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


def wavefront_x_vs_t(Nsite, Nlayer, mydata):
    perc = 0.04
    def sum_till_n(n, pL):
        return np.sum(pL[:n])
    Sp = np.vectorize(sum_till_n, excluded=['pL'])
    pX = mydata[:, :Nsite * Nlayer]
    pX = (pX + pX[:, ::-1])/2.0
    x = np.zeros(len(pX[:,0]))  

    for t in range(len(pX[:,0])):

        pL = pX[t].reshape(Nlayer, Nsite) 
        pL = np.sum(pL, axis=0)
        pL = pL / np.sum(pL)
        # sum till n site

        Ip = Sp(n = np.arange(Nsite//2), pL = pL[:])
        x[t] = np.argmin(np.abs(Ip - perc))

    return x

#fold = [f"brun{i}" for i in range(vnum)]
vnum = 0

Vgs = []
Egs = []
for nl in nl_list:
    for γ, Rsample in zip(γ_list, Rsample_list):
        Vg_ = []
        Eg_ = []
        for E0, dE in zip(E0_list, dE_list):
            Nread = 0
            Layersaddupdata = []
            for R in range(Rsample):
                ############################################
                Layersaddupfile = os.path.isfile('./brun'+str(vnum)+'/output/Layersaddup-'+methodfile+'-'+modelfile+'.npy')
                if Layersaddupfile == True:
                    print('./brun'+str(vnum)+'/output/*****-'+methodfile+'-'+modelfile+'.npy being read!!')
                    Layersaddup = (np.load('./brun'+str(vnum)+'/output/Layersaddup-'+methodfile+'-'+modelfile+'.npy')).T
                    ntraj = Layersaddup.shape[0]
                    #print("ntraj: ", ntraj)
                    for itraj in range(ntraj):
                        Layersaddupdata.append(Layersaddup[itraj,:, :])
                        Nread += 1
                vnum+=1
            if Nread == 0:
                parnum += 1
                continue
            #print("Layersaddupdata shape: ", np.array(Layersaddupdata).shape)
            os.system('mkdir -p parnum'+str(parnum))
            Layerdataavg = np.mean(np.array(Layersaddupdata), axis=0)
            #print("Layerdataavg shape: ", Layerdataavg.shape)
            np.save('parnum'+str(parnum)+'/Layersaddup-'+methodfile+'-'+modelfile+'_'+str(parnum)+'.npy', Layerdataavg)
            print("parnum ", parnum,", Nsite=",Nsite,", nl=", nl,", γ=",γ,", E0=",E0,", dE=",dE," and Disorder #=",Nread,"done!")
            #####
            y = wavefront_x_vs_t(Nsite, 1, Layerdataavg[:, :Nsite])*a /Å
            x = np.arange(0, NSteps, Nskip) * dtN/ps
            #y = x_ts * a /Å
            print(x.shape, y.shape)
            slope, yint = linregress(x[9:-1], y[9:-1])[:2]
            Vg_.append(slope)
            Eg_.append(E0)
            #####
            parnum += 1
        Vgs.append(np.array(Vg_))
        Egs.append(np.array(Eg_))
        if len(Vg_) != 0:
            np.save("Vg_γ"+str(γ)+"_nl"+str(nl)+".npy", abs(np.array(Vg_)))
            np.save("Eg_γ"+str(γ)+"_nl"+str(nl)+".npy", np.array(Eg_))

