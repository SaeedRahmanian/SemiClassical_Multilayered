import sys, os
import numpy as np
import time

# Path to builder directory
current_directory = os.getcwd()
HOME = os.path.join(*current_directory.split('/')[:4])#"/scratch/user/u.sr215595/"
Project = str(sys.argv[1])
JobName = str(sys.argv[2])
print(f"Project: {Project}, ", f"JobName: {JobName}")
LOCAL_DIR=os.getcwd()#HOME + "Projects/" + Project + "/build/" + JobName
#ScratchFolder = HOME + "/scratch/" # Directory you save the data

print("Serial directory: ", os.system("pwd"))

modelfile = str(sys.argv[3])
methodfile = str(sys.argv[4])                                                                                               
Nlayer = int(sys.argv[5])                                                                                                                                                                                             
NSteps = int(sys.argv[6])                                                                                                                                                                          
E0 = float(sys.argv[7])  
dE = float(sys.argv[8])                                                                                            
NTraj = int(sys.argv[9])                                                                                             
γ = str(sys.argv[10])
jobnum = int(sys.argv[11])
# Nsite = int(sys.argv[11])                                                                                                                                                                                              
# dtN = float(sys.argv[12])                                                                                                                                                                            
# Nskip = int(sys.argv[13])                                                
# dE = float(sys.argv[14]) 
# branch = int(sys.argv[15])                                                                                              
# a = str(sys.argv[16]) 


# Read the contents of the Model, Method and input files
Model_contents=open(LOCAL_DIR + f"/Model/{modelfile}.py",'r').read()
Method_contents=open(LOCAL_DIR + f"/Method/{methodfile}.py",'r').read()
input_contents=open(f"./input.txt",'r').read()


# Edit model file
fout=open(f"./Model/{modelfile}.py",'w')
contents=Model_contents.replace('***Nlayer***', str(Nlayer))
contents=contents.replace('***NSteps***', str(NSteps))
contents=contents.replace('***E0***',str(E0))
contents=contents.replace('***dE***',str(dE))
contents=contents.replace('***NTraj***', str(NTraj))
contents=contents.replace('***γ***', str(γ))
fout.write(contents)
fout.close()



# Edit Method file
fout=open(f"./Method/{methodfile}.py",'w')
contents=Method_contents.replace('***jobnum***', str(int(jobnum)))
fout.write(contents)
fout.close()

# Edit Input file
fout=open(f"./input.txt",'w')
contents=input_contents.replace('***methodfile***', methodfile)
fout.write(contents)
fout.close()

# The path to the Model and Method folders
sys.path.append(os.popen("pwd").read().replace("\n","")+"/Method")
sys.path.append(os.popen("pwd").read().replace("\n","")+"/Model")


#-------------------------
print("Reading input.txt")
inputtxt = open('input.txt', 'r').readlines()


def getInput(input,key):
    try:
        txt = [i for i in input if i.find(key)!=-1][0].split("=")[1].split("#", 1)[0].replace("\n","")
    except:
        txt = ""
    return txt.replace(" ","")

model_ =  getInput(inputtxt,"Model")
method_ = getInput(inputtxt,"Method").split("-")
exec(f"import {model_} as model")
exec(f"import {method_[0]} as method")
try:
    stype = method_[1]
except:
    stype = "_"
#-------------------------

fold = "./output"

NTraj = model.parameters.NTraj
NStates = model.parameters.NStates

#------ Arguments------------------
par = model.parameters() 
print("par was read, γ=", par.γ)
par.ID     = np.random.randint(0,100)
par.SEED   = np.random.randint(0,100000000)
    
#---- methods in model ------
#par.dHel = model.dHel
par.dHel0 = model.dHel0
par.initR = model.initR
par.HelR   = model.HelR
par.HelL   = model.HelL
par.Hel0   = model.Hel0
par.stype = stype

if method_[0]=="nrpmd":
    par.initHel0 = model.initHel0
    

#---- overriden parameters ------

parameters = [i for i in inputtxt if i.split("#")[0].split("=")[0].find("$") !=- 1]
for p in parameters:
    exec(f"par.{p.split('=')[0].split('$')[1]} = {p.split('=')[1].split('#')[0]}")
    print(f"Overriding parameters: {p.split('=')[0].split('$')[1]} = {p.split('=')[1].split('#')[0]}")
#--------------------------------

t1 = time.time()
#------------------- run --------------- 
#rho_sum, psi_sum  = method.runTraj(par)
rho_sum, psi_sum, Layersaddup, wavefront, MSD_exc = method.runTraj(par)
#--------------------------------------- 

try:
    PiiFilename =  f"{fold}/{method_[0]}-{method_[1]}-{model_}"
    psiFilename =  f"{fold}/psi-{method_[0]}-{method_[1]}-{model_}.npy"  
    LayersaddupFilename =  f"{fold}/Layersaddup-{method_[0]}-{method_[1]}-{model_}.npy"
    wavefrontFilename =  f"{fold}/wavefront-{method_[0]}-{method_[1]}-{model_}.npy"
    MSD_excFilename =  f"{fold}/MSD_exc-{method_[0]}-{method_[1]}-{model_}.npy"
    #PiiFile = open(PiiFilename,"w+") 
except:
    PiiFilename = f"{fold}/{method_[0]}-{model_}"
    psiFilename = f"{fold}/psi-{method_[0]}-{model_}.npy"
    LayersaddupFilename = f"{fold}/Layersaddup-{method_[0]}-{model_}.npy"
    wavefrontFilename = f"{fold}/wavefront-{method_[0]}-{model_}.npy"
    MSD_excFilename = f"{fold}/MSD_exc-{method_[0]}-{model_}.npy"
    #

NTraj = par.NTraj

#times = np.arange(rho_sum.shape[-1]) * par.nskip * par.dtN, par.nskip * par.dtN

if (method_[0] == 'sqc'):
    PiiFile = open(PiiFilename,"w+") 
    for t in range(rho_sum.shape[-1]):
        PiiFile.write(f"{t * par.nskip * par.dtN} \t")
        norm = 0
        for i in range(NStates):
            norm += rho_sum[i,t].real
        for i in range(NStates):
            PiiFile.write(str(rho_sum[i,t].real / ( norm ) ) + "\t")
        PiiFile.write("\n")
    PiiFile.close()
    
else:
    # rho_sum[:, 1:] = rho_sum[:, 1:]#/ NTraj
    # for i in range(rho_sum.shape[2]):
    #     np.savetxt(PiiFilename+'_traj'+str(i)+'.txt', (rho_sum[:,:,i]).T.real)
    #np.save(psiFilename, psi_sum)
    np.save(LayersaddupFilename, Layersaddup)
    #np.save(wavefrontFilename, wavefront)
    #np.save(MSD_excFilename, MSD_exc)
    #print("psi_sum:\n", psi_sum)


# def MSD(NStates, a, rhodiag):
#     Matrix_r = np.arange(L)*a
#     return sum(rhodiag*(np.arange(L)**2)*(a**2) - (rhodiag*np.arange(L)*a)**2)


t2 = time.time()-t1
print(f"Total Time: {t2}")
print(f"Time per trajectory: {t2/NTraj}")