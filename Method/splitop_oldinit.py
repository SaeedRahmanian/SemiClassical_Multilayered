import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pylab as plt
# jit
import time
# from numba import jit, objmode

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


def Hlayer(myk, parameters, dat):
    Nlayer = parameters.Nlayer
    az = parameters.az
    εk = parameters.εk
    g  = parameters.g
    kz = parameters.kz
    ωk = parameters.ωk
    ω0 = parameters.ω0
    Lz = parameters.Lz
    Rlz = parameters.Rlz
    Hk = np.zeros((Nlayer + 1, Nlayer + 1))
    for nl in range(Nlayer):
        Hk[nl,nl] = εk[myk]
        Hk[nl, -1] = g * np.sqrt(ωk[myk]/ω0) * np.sin(kz*((Rlz[nl] * az) + Lz/2))
        Hk[-1, nl] = g * np.sqrt(ωk[myk]/ω0) * np.sin(kz*((Rlz[nl] * az) + Lz/2))
    Hk[-1,-1] = ωk[myk]
    myEk, myVk = np.linalg.eigh(Hk)
    return myEk, myVk


def EW2KGauss(parameters, dat):
    ### Split operator method
    Nsite = parameters.Nsite
    Npht = parameters.Nsite
    Nlayer = parameters.Nlayer
    ke = parameters.ke
    Rn = parameters.Rn
    a = parameters.a
    mask = parameters.mask
    μg = 1 * Nsite//2 # to control wather center of Gaussian is at the center or at teh edge


    σk = ***σk***
    ΔE = ***ΔE***
    dE = ΔE/27.2114
    Emin = parameters.Emin
    E0n = ***E0n***
    E0 = Emin + E0n * dE
    print("Emin = ", Emin, ", E0-dE = ", E0-dE, ", E0+dE = ", E0+dE)
    print("Creating initial state, E0 - dE=", E0-dE, ", E0 + dE=", E0 + dE, ", dE=", dE)
    
    ######################
    ckp = np.zeros(Nlayer * Nsite + Npht)
    ckm = np.zeros(Nlayer * Nsite + Npht)
    kp, km = [], []

    print("Finding coefficients")
    for i in range(Nsite//2, Nsite):
        Ek, Vk = Hlayer(i, parameters, dat)
        #print("Elow = ", E0 - dE, "E = ", Ek[0] , "E upper = ", E0 + dE)
        if Ek[0]> E0-dE and Ek[0]< E0 + dE and a*ke[i] >= np.pi:
            km.append(i)
            kp.append(Nsite - 1 - i)
            for nl in range(Nlayer):
                ckm[nl * Nsite + i] = Vk[nl,0]
                ckp[nl * Nsite - i + (Nsite - 1)] = Vk[nl,0]
            ckm[Nlayer * Nsite + i] = Vk[-1,0]
            ckp[Nlayer * Nsite - i + (Nsite - 1)] = Vk[-1,0]

    print("kp = ", kp)
    print("km = ", km)
    kp = kp[::-1]
    kp0 = np.mean([ke[j] for j in kp])
    km0 = np.mean([ke[j] for j in km])

    print("Creating initial state")
    cnp = np.zeros(Nlayer * Nsite + Npht, dtype=complex)
    cnm = np.zeros(Nlayer * Nsite + Npht, dtype=complex)
    for nl in range(Nlayer):
        for i in range(Nsite):
            for j, l in zip(kp, km):
                cnp[nl * Nsite + i] += (np.exp(1.j*μg*ke[j]*a))*np.exp(1.j*Rn[i]*ke[j]*a) * ckp[nl * Nsite + j] * np.exp(-(a*(ke[j]-kp0))**2/(2*(σk**2)))/(np.sqrt(Nsite))
                cnm[nl * Nsite + i] += (np.exp(1.j*μg*ke[l]*a))*np.exp(1.j*Rn[i]*ke[l]*a) * ckm[nl * Nsite + l] * np.exp(-(a*(ke[l]-km0))**2/(2*(σk**2)))/(np.sqrt(Nsite))
    for j, l in zip(kp, km): 
        cnp[Nlayer * Nsite + j] = (np.exp(1.j*μg*ke[j]*a))*ckp[Nlayer * Nsite + j] * np.exp(-(a*(ke[j]-kp0))**2/(2*(σk**2))) 
        cnm[Nlayer * Nsite + l] = (np.exp(1.j*μg*ke[l]*a))*ckm[Nlayer * Nsite + l] * np.exp(-(a*(ke[l]-km0))**2/(2*(σk**2)))

    c = cnp / np.sqrt(sum(abs(cnp)**2)) + cnm / np.sqrt(sum(abs(cnm)**2))
    c = c / np.sqrt(sum(abs(c)**2))

    cmasked = np.zeros(Nlayer * Nsite + parameters.N_Pht) + 0.j
    cmasked[:Nlayer * Nsite] = c[:Nlayer * Nsite]
    cmasked[Nlayer * Nsite:] = (c[Nlayer * Nsite:])[mask]
    cmasked = cmasked / np.sqrt(sum(abs(cmasked)**2))

    # No need to copy for job running
    selectk = []
    for i in kp:
        selectk.append(np.exp(-(a*(ke[i]-kp0))**2/(2*(σk**2)))) #/ (σk * np.sqrt(2*np.pi)))
    for i in km:
        selectk.append(np.exp(-(a*(ke[i]-km0))**2/(2*(σk**2)))) #/ (σk * np.sqrt(2*np.pi)))
    np.save("ks_gaussian.npy", selectk)
    return cmasked


# Generating the initial state of the electronic part:
def initElectronic(parameters, dat, initState = 0):
    initype = parameters.initype
    Nsite = parameters.Nsite
    Nlayer = parameters.Nlayer
    if initype == 5:
        c = EW2KGauss(parameters, dat)
    print("Exciton population: ", sum(abs(c[:Nsite * Nlayer])**2))
    print("Photon population: ", sum(abs(c[Nsite * Nlayer:])**2))
    print("c size = ", len(c))
    #print("shapes: ", (dat.Hij).shape, (dat.Hij0).shape)
    #print("State energy: ", np.conjugate(c.T) @ (np.diag(dat.Hij) + dat.Hij0) @ c)
    print("c normalization = ", sum(abs(c)**2))
    print("max(abs(c)): ", max(abs(c)))
    if max(abs(c))==0 or max(abs(c))==float('nan'):
        c[initState] = 1
        Print("No state where found")
        exit()
    return c

def Enoncouplefunc(parameters):
    Nsite = parameters.Nsite
    Nlayer = parameters.Nlayer
    Npht = parameters.N_Pht
    εk = parameters.εk
    mask = parameters.mask
    mirrormask = list(map(bool, 1-mask*1))
    Enoncouple = np.zeros(Nlayer * Nsite + Npht)
    for nl in range(Nlayer):
        Enoncouple[nl*(Nsite - Npht): (nl+1)*(Nsite - Npht)] = εk[mirrormask]
    return Enoncouple

def Ushuf(parameters, ci): #This unitary rotates state  to the block digonal order
    N = parameters.Nsite
    Nlayer = parameters.Nlayer
    Npht = parameters.N_Pht
    mask = parameters.mask
    mirrormask = list(map(bool, 1-mask*1))
    # This ci is for test
    ###ci = np.arange(N*Nlayer)
    ###ci = np.concatenate((ci, np.arange(N*Nlayer, N*Nlayer + N)[mask]))
    #print("\nci: \n", ci)
    ci_sh = np.reshape(ci[:N * Nlayer], (Nlayer, N)).T
    #print("msk:", mask)
    ci_sh_int = (ci_sh[mask, :]).T
    ci_sh_int = np.vstack((ci_sh_int, ci[N * Nlayer:]))
    ci_sh_noint = ci_sh[mirrormask, :]
    cishuf = np.concatenate(((ci_sh_noint.T).flatten(), (ci_sh_int.T).flatten()))
    #print("cishuf: \n", cishuf)
    return cishuf


def Ushufrev(parameters, ci): #This unitary rotates state back to the non-block digonal order
    N = parameters.Nsite
    Nlayer = parameters.Nlayer
    Npht = parameters.N_Pht
    mask = parameters.mask
    mirrormask = list(map(bool, 1-mask*1))
    ci_sh_noint = np.reshape(ci[:Nlayer*(N - Npht)], (Nlayer, N-Npht)).T
    ci_sh_int = np.reshape(ci[Nlayer*(N - Npht):], (Npht, Nlayer+1))
    # print("ci_sh_int = \n", ci_sh_int)
    # print("ci_sh_noint = \n", ci_sh_noint)
    cireshuf = np.zeros((N, Nlayer), dtype=complex)
    cireshuf[mask, :] = ci_sh_int[:,:-1]
    cireshuf[mirrormask, :] = ci_sh_noint[:,:]
    cireshuf = np.concatenate(((cireshuf.T).flatten(), (ci_sh_int[:,-1]).flatten()))
    #print("cireshuf compare: ", max(abs(cireshuf-np.arange(N*Nlayer+Npht))))
    return cireshuf


# Split operator function
def splitEvolution(parameters, ε0, E2, V2, dt, ct):
    Nlayer = parameters.Nlayer
    Rlz = parameters.Rlz
    Lz = parameters.Lz
    kz = parameters.kz
    az = parameters.az
    ω0 = parameters.ω0
    NormN = parameters.NormN

    tlc2 = np.zeros(2, dtype=complex)

    # Transforming the basis to the new single state coupled to the cavity
    tlc2[0] = sum([cm * sinm for cm, sinm in zip(ct[:-1], np.sin(kz*((Rlz * az) + Lz/2))/NormN)]) # ∑m cm * cos(kz * Rzm) 
    tlc2[1] = ct[-1] # Ph0ton state which remains unchanged

    # Applying V2 operaor 
    tlc2p = np.dot(V2, tlc2)

    # applying the exponential matrix
    tlc2p_E =  np.exp(-1.j * E2 * dt) * tlc2p

    # Applying complex conjugate of the V2 operaor 
    tlc2p_p = np.dot(np.conjugate(V2.T), tlc2p_E)

    Δc = tlc2p_p[0] - (tlc2[0] * np.exp(-1.j * ε0 * dt))
    cdt = np.zeros(Nlayer+1, dtype=complex)
    cdt[:-1] = cdt[:-1] + ct[:-1] * np.exp(-1.j * ε0 * dt)
    cdt[:-1] = cdt[:-1] + Δc * np.sin(kz * ((Rlz * az) + Lz/2))/NormN
    cdt[-1] = tlc2p_p[-1] 
    return cdt


def BlocksED(parameters):
    Nsite = parameters.Nsite
    ω0 = parameters.ω0
    ωk = parameters.ωk
    εk = parameters.εk
    g = parameters.g
    NormN = parameters.NormN
    mask = parameters.mask
    E2s = []
    V2s = []
    for n in range(Nsite):
        if mask[n] == True:
            H2 = np.zeros((2,2), dtype=complex)
            H2[0,0] = εk[n]
            H2[1,1] = ωk[n]
            H2[0,1], H2[1,0] = np.sqrt(ωk[n]/ω0) * g * NormN,    np.sqrt(ωk[n]/ω0) * g * NormN
            E2,V2 = np.linalg.eigh(H2)
            E2s.append(E2)
            V2s.append(V2)
    return E2s, V2s


def propagateCi(parameters, dat, ci, dt):
    tp0 = time.time()
    Enoncouple = dat.Enoncouple
    Nsite = parameters.Nsite
    Nlayer = parameters.Nlayer
    Npht = parameters.N_Pht
    εk = parameters.εk
    Ndiag = (Nsite - Npht) * Nlayer

    ct_trot = ci + 0.j
    ct_trot = np.exp(-1.j * 0.5 * dt * dat.Hij) * ct_trot
    
    tfft0 = time.time()
    for nl in range(Nlayer):
        ct_trot[Nsite*nl:Nsite*(nl+1)] = fft(ct_trot[Nsite*nl:Nsite*(nl+1)], norm = 'ortho')
    tfft1 = time.time()

    tsh0 = time.time()
    ct_shuf = ct_shuf = Ushuf(parameters, ct_trot)
    tsh1 = time.time()

    tev0 =  time.time()
    ct_shuf = np.exp(-1.j * dt * Enoncouple) * ct_shuf
    for p in range(Npht):
        ct_shuf[Ndiag + p*(Nlayer+1): Ndiag + (p+1)*(Nlayer+1)] = splitEvolution(parameters, εk[dat.Pht_ks[p]], dat.E2_list[p], dat.V2_list[p], dt, ct_shuf[Ndiag + p*(Nlayer+1): Ndiag + (p+1)*(Nlayer+1)])
    tev1 =  time.time()
    print("Evolution = ", (tev1-tev0))

    tsh2 = time.time()
    ct_trot = Ushufrev(parameters, ct_shuf)
    tsh3 = time.time()
    print("Shufling = ", (tsh1-tsh0) )
    print("ReShufling = ", (tsh3-tsh2))

    tfft2 = time.time()
    for nl in range(Nlayer):
        ct_trot[Nsite*nl:Nsite*(nl+1)] = ifft(ct_trot[Nsite*nl:Nsite*(nl+1)], norm = 'ortho')
    tfft3 = time.time()
    print("Fourier transform = ", (tfft1-tfft0) + (tfft3-tfft2))

    ct_trot = np.exp(-1.j * 0.5 * dt * dat.Hij) * ct_trot
    print(">>>>> Propagate time = ", time.time() - tp0)
    return ct_trot


#@jit(nopython=False)
def Force(γ, dH0, ci):

    # dH = dat.dHij #dHel(R) # Nxnxn Matrix, N = Nuclear DOF, n = NStates 
    # dH0  = dat.dH0 
    # ci = dat.ci

    F = -dH0 #np.zeros((len(dat.R)))
    #F -= np.real(np.einsum('ijk,i,j->k', dHij, ci.conjugate(), ci))
    F -= γ * (ci.conj() * ci).real
    # for i in range(len(ci)):
    #     #print("---", F.shape, dHij.shape, ci.shape)
    #     F -= dHij[i,i,:]  * (ci[i] * ci[i].conjugate() ).real
    #     for j in range(i+1, len(ci)):
    #         F -= 2.0 * dHij[i,j,:]  * (ci[i].conjugate() * ci[j] ).real

    return F

def VelVer(dat) : 
    par =  dat.param
    v = dat.P/par.M
    F1 = dat.F1 
    # electronic wavefunction
    ci = dat.ci * 1.0
    
    EStep = int(par.dtN/par.dtE)
    dtE = par.dtN/EStep

    # half electronic evolution
    for t in range(int(np.floor(EStep/2))):
        ci = propagateCi(par, dat, ci, dtE)#propagateCi(ci, dat.E0, dat.V0, dat.Hij, dtE)  
    ci /= np.sum(ci.conjugate()*ci) 
    dat.ci = ci * 1.0 

    # ======= Nuclear Block ==================================
    #print("v:\n", dat.P, par.M)
    #print("Fi:\n", F1)
    dat.R += v * par.dtN + 0.5 * F1 * par.dtN ** 2 / par.M
    
    #------ Do QM ----------------
    #dat.Hij  = par.Hel() + 0j
    
    dat.Hij  = par.HelR(dat.R)  
    #dat.dHij = par.dHel(dat.R)
    dat.dH0  = par.dHel0(dat.R)
    #-----------------------------
    F2 = Force(par.γ, dat.dH0, dat.ci) # force at t2
    v += 0.5 * (F1 + F2) * par.dtN / par.M
    dat.F1 = F2
    dat.P = v * par.M
    # ======================================================
    # half electronic evolution
    for t in range(int(np.ceil(EStep/2))):
        ci = propagateCi(par, dat, ci, dtE) 
    ci /= np.sum(ci.conjugate()*ci)  
    dat.ci = ci * 1.0 

    return dat


def pop(dat):
    ci =  dat.ci
    return np.outer(ci.conjugate(),ci)

def runTraj(parameters):
    #------- Seed --------------------
    try:
        np.random.seed(parameters.SEED)
    except:
        pass
    #------------------------------------
    ## Parameters -------------
    dat = Bunch(param =  parameters )
    NSteps = parameters.NSteps
    NTraj = parameters.NTraj
    Nsite = parameters.Nsite
    Nlayer = parameters.Nlayer
    Npht = parameters.N_Pht
    NStates = parameters.NStates
    initState = parameters.initState # intial state
    nskip = parameters.nskip
    a = parameters.a
    ωk = parameters.ωk
    mask = parameters.mask
    ke = parameters.ke
    kph = ke[mask]
    Rn = parameters.Rn
    # Electronic Hamiltonian with no disorder

    
    #---------------------------
    if NSteps%nskip == 0:
        pl = 0
    else :
        pl = 1
    rho_ensemble = np.zeros((Nsite * Nlayer + Npht, NSteps//nskip + pl), dtype=complex)
    # Ensemble
    #dat.Hij0 = parameters.Hel0()
    #dat.E0, dat.V0 = np.linalg.eigh(dat.Hij0)
    dat.Enoncouple = Enoncouplefunc(parameters)
    dat.E2_list, dat.V2_list = BlocksED(parameters)
    dat.Pht_ks = np.arange(Nsite)[mask]
    tinit0 = time.time()
    # Call function to initialize mapping variables
    dat.ci = initElectronic(parameters, dat, initState = 0) # np.array([0,1])
    print("Initial state:", time.time() - tinit0)
    for itraj in range(NTraj): 
        # Trajectory data
        dat.R, dat.P = parameters.initR()
        
        # set propagator
        vv  = VelVer

        #----- Initial QM --------
        dat.Hij  = parameters.HelR(dat.R)  
        #dat.dHij = parameters.dHel(dat.R) 
        dat.dH0  = parameters.dHel0(dat.R)
        dat.F1 = Force(parameters.γ, dat.dH0, dat.ci) # Initial Force
        #----------------------------
        iskip = 0 # please modify
        t0 = time.time()
        for i in range(NSteps): # One trajectory
            #------- ESTIMATORS-------------------------------------
            if (i % nskip == 0):
                Eikn = np.exp(-1.j * np.outer(Rn, kph * a))/np.sqrt(Nsite)
                cph = fft(dat.ci[Nsite * Nlayer:], norm = 'ortho')
                rho_ensemble[:Nsite * Nlayer,iskip] += np.abs(dat.ci[:Nsite * Nlayer])**2 
                rho_ensemble[Nsite * Nlayer:,iskip] += np.abs(cph)**2 
                iskip += 1
            #-------------------------------------------------------
            dat = vv(dat)
        time_taken = time.time()-t0
        print(f"Time taken: {time_taken} seconds")


    return rho_ensemble