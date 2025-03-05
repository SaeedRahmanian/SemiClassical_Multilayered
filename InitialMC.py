#!/sw/eb/sw/Anaconda3/2024.02-1/bin/python

##NECESSARY JOB SPECIFICATIONS
#SBATCH --partition=shared
#SBATCH --job-name=Inisialstate         #Set the job name to "JobExample1"
#SBATCH --time=0-05:00:00          #Set the wall clock limit to 1hr and 30min
#SBATCH --ntasks=1               #Request 1 task
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G              
#SBATCH --output=output/out.%j       

import numpy as np
import matplotlib.pylab as plt
from scipy.fft import fft, ifft
import sys, os

sys.path.append(os.popen("pwd").read().replace("\n","")+"/Model")

#-------------------------
try:
    inputtxt = open(sys.argv[1], 'r').readlines()
    print(f"Reading {sys.argv[1]}")
except:
    print("Reading input.txt")
    inputtxt = open('input.txt', 'r').readlines()


def getInput(input,key):
    try:
        txt = [i for i in input if i.find(key)!=-1][0].split("=")[1].split("#", 1)[0].replace("\n","")
    except:
        txt = ""
    return txt.replace(" ","")

model_ =  getInput(inputtxt,"Model")
exec(f"import {model_} as model")
#-------------------------

os.system("mkdir -p InitialStates")

par = model.parameters() 

def Epol(param):
    Nsite = param.Nsite
    def Hlayer(i):
        h22 = np.zeros((2,2))
        h22[0,0] = param.εk[i]
        h22[1,1] = param.ωk[i]
        NormN = param.NormN
        h22[0,1] = param.g * np.sqrt(param.ω0/param.ωk[i]) * NormN
        h22[1,0] = h22[0,1]
        return np.linalg.eigh(h22)


    U = np.zeros((2, Nsite), dtype='complex128')
    Epl = np.zeros((Nsite), dtype='complex128')
    for i in range(Nsite):
        Ek, Vk = Hlayer(i)
        U[:,i] = Vk[:, param.branch]
        Epl[i] = Ek[param.branch]
    return Epl, U

def initElectronicOPT(param, ntry = 50000):
    """ 
    branch = 0 means LP
    branch = -1 means UP
    """
    branch = param.branch
    E0= param.E0
    dE = param.dE
    Nsite = param.Nsite
 
 
    k = param.ke
    Epl, U = Epol(param)
    # Finding the initial state
    idx = np.arange(len(Epl))[(Epl>E0 - dE/2) & (Epl< E0 + dE/2)] 

    ns = len(idx)
    print("number of initial states = ", ns)

    a = np.ones((ns))/ns**0.5 + 0j

    def minimizeF(c) :
        N = len(c)
        n = np.arange(len(c)) - N//2
        return np.sum(n**2 * c.conj() * c).real

    u = np.exp(1j * np.outer(k[idx] * param.a, np.arange(Nsite)))
    c =  np.einsum('kj, k -> j', u, a)  
    c = c/np.sum(c.conj() * c)**0.5
    accept = 0
    x2old = minimizeF(c)
    ax, fig = plt.subplots()
    for i in range(ntry):   
        a2 = a * 1.0
        θ = np.random.random() * 2 * np.pi
        A = (np.random.random() - 0.5)
        a2[np.random.choice(np.arange(len(a)))] += A * np.exp(1j * θ)
        a2 = (a2 + a2[::-1])
        a2 = a2/np.sum(np.abs(a2)**2)**0.5

        # cthis =   np.einsum('kj, k -> j', u, a2)  
        # cthis = cthis/np.sum(cthis.conj() * cthis)**0.5
        # x2new = minimizeF(cthis)
        
        aOPT = np.zeros((Nsite), dtype='complex128')
        aOPT[idx] = a2
        Cik = np.einsum('ik, k -> ik', U, aOPT) 
        ψex = np.fft.ifft(Cik[0,:], norm='ortho')
        x2new = minimizeF(ψex)
        
        if x2new < x2old:
            a = a2 * 1.0
            x2old = x2new
            accept += 1
            #print(accept)
            if accept % 10 == 0:
                
                plt.scatter(i, np.log(x2new), c ='black')
 
                print("x2 = ", x2old)
            
    c = ψex
    aFull =  np.zeros((Nsite), dtype='complex128')
    aFull[idx] = a
    plt.savefig(f'./InitialStates/Convergance_E0={param.E0 / param.eV:.2f}_dE={int(param.dE / param.eV * 1000)}.png')
    ax, fig = plt.subplots()
    plt.plot(abs(ψex)**2, 'r')
    plt.suptitle('Photon')
    plt.savefig(f'./InitialStates/Photonabswf_E0={param.E0 / param.eV:.2f}_dE={int(param.dE / param.eV * 1000)}.png')
    print("exciton state saved")
    ax, fig = plt.subplots()
    plt.plot(abs(np.fft.ifft(Cik[1,:], norm='ortho'))**2, 'r')
    plt.suptitle('Exciton')
    plt.savefig(f'./InitialStates/Excitonabswf_E0={param.E0 / param.eV:.2f}_dE={int(param.dE / param.eV * 1000)}.png')
    print("Phootn state saved")
    return aFull,  c

def initElectronic(param, OPT = True, c = 0, a = 0):
    if OPT:
        a, c = initElectronicOPT(param, param.ntry)

        np.save(f'./InitialStates/InitGentor_E0={param.E0 / param.eV:.2f}_dE={int(param.dE / param.eV * 1000)}.npy', a)

    nl = param.Nlayer
    _, U = Epol(param)
    
    Cik = np.einsum('ik, k -> ik', U, a) 
    ψ = Cik.flatten()
    
    ψnew = ψ * 0.0    
    ψnew[: param.Nsite] = np.fft.ifft(ψ[:param.Nsite], norm='ortho')
    ψnew[param.Nsite:] = ψ[param.Nsite:]
    # expand to multilayer
    sinkm = np.sin(param.kz * ((param.Rlz * param.az) + param.Lz/2))

    ψml = np.zeros(((nl + 1) * param.Nsite), dtype='complex128')
    for i in range(nl):
        ψml[i * param.Nsite: (i+1) * param.Nsite] =  (sinkm[i]/param.NormN) * ψnew[:param.Nsite] 
    ψml[nl * param.Nsite:] = ψnew[param.Nsite:]  
    return ψml, c

if sys.argv[1] == "Findwfseed":
    a, c = initElectronicOPT(par, par.ntry)
    np.save(f'./InitialStates/InitGentor_E0={par.E0 / par.eV:.2f}_dE={int(par.dE / par.eV * 1000)}.npy', a)
if sys.argv[1] == "Findwf":
    if os.path.isfile(f'./InitialStates/InitGentor_E0={par.E0 / par.eV:.2f}_dE={int(par.dE / par.eV * 1000)}.npy') == True:
        print("--------------- \n Generator exist\n ---------------")
        dat = np.load(f'./InitialStates/InitGentor_E0={par.E0 / par.eV:.2f}_dE={int(par.dE / par.eV * 1000)}.npy')
        ψ, c  = initElectronic(par, par.ntry, OPT = False, a = dat) 
        np.save(f'./InitialStates/InitState_Nlayer={par.Nlayer}_E0={par.E0 / par.eV:.2f}_dE={int(par.dE / par.eV * 1000)}.npy', ψ)
    if os.path.isfile(f'./InitialStates/InitGentor_E0={par.E0 * par.eV:.2f}_dE={int(par.dE / par.eV * 1000)}.npy') == False:
        print("--------------- \n Generator doesn't exist\n ---------------")
        ψ, c  = initElectronic(par, par.ntry, OPT = True) 
        np.save(f'./InitialStates/InitState_Nlayer={par.Nlayer}_E0={par.E0 / par.eV:.2f}_dE={int(par.dE / par.eV * 1000)}.npy', ψ)

    ψmask = np.zeros(par.Nsite*par.Nlayer + par.N_Pht, dtype=complex)
    ψmask[: par.Nsite * par.Nlayer] = (ψ[ : par.Nsite * par.Nlayer])
    ψmask[par.Nsite * par.Nlayer:] = (ψ[par.Nsite*par.Nlayer:])[par.mask]
    np.save(f'./InitialStates/InitState_E0={par.E0 / par.eV:.2f}_dE={int(par.dE / par.eV * 1000)}.npy', ψmask)

    fig, ax = plt.subplots()
    plt.plot(abs(ψmask[: par.Nsite * par.Nlayer])**2)
    plt.savefig(f'./InitialStates/wfexciton_E0={par.E0 / par.eV:.2f}_dE={int(par.dE / par.eV * 1000)}.png')

    fig, ax = plt.subplots()
    plt.plot(abs(abs(fft(ψmask[par.Nsite * par.Nlayer : ], norm='ortho'))**2)**2)
    plt.savefig(f'./InitialStates/wfphoton_E0={par.E0 / par.eV:.2f}_dE={int(par.dE / par.eV * 1000)}.png')