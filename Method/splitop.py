import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pylab as plt
# jit
import time
# from numba import jit, objmode

class Bunch:
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


# This function used in split operator method to find the coefficient for each k-points
def Hlayer_v1(myk, parameters, dat):
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

def Hlayer_v2(myk, parameters, dat):
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
        Hk[nl, -1] = g * np.sqrt(ω0/ωk[myk]) * np.sin(kz*((Rlz[nl] * az) + Lz/2))
        Hk[-1, nl] = g * np.sqrt(ω0/ωk[myk]) * np.sin(kz*((Rlz[nl] * az) + Lz/2))
    Hk[-1,-1] = ωk[myk]
    myEk, myVk = np.linalg.eigh(Hk)
    return myEk, myVk


# This function creates Gaussian initial state for the electronic part and Photonic part (The Gaussian is in real space)
def EW2KGauss(parameters, dat):
    ### Split operator method
    Nsite = parameters.Nsite
    Npht = parameters.Nsite
    Nlayer = parameters.Nlayer
    ke = parameters.ke
    Rn = parameters.Rn
    a = parameters.a
    mask = parameters.mask
    eV = parameters.eV
    μg = 1 * Nsite//2 # to control wather center of Gaussian is at the center or at teh edge
    jp = 1; jm = 1 # whether to choose positive or negative k
    pol = np.array(["lower", "upper"])[parameters.branch] # "lower" or "upper"

    σk = 0.0005
    dE = parameters.dE
    Emin = parameters.Emin
    E0 = parameters.E0
    print("Emin = ", Emin, ", E0-dE = ", E0-dE, ", E0+dE = ", E0+dE)
    print("Creating initial state, E0 - dE=", E0-dE, ", E0 + dE=", E0 + dE, ", dE=", dE)
    
    ######################
    ckp = []#np.zeros(Nlayer * Nsite + Npht)
    ckm = []#np.zeros(Nlayer * Nsite + Npht)
    dkp = np.zeros(Nsite)
    dkm = np.zeros(Nsite)
    kp, km = [], []

    tinit0 = time.time()
    if pol == "lower":
        print("Finding coefficients for lower polariton")
        for i in range(Nsite//2, Nsite):
            Ek, Vk = Hlayer_v1(i, parameters, dat)
            #print("Elow = ", E0 - dE, "E = ", Ek[0] , "E upper = ", E0 + dE)
            if Ek[0]> E0-dE and Ek[0]< E0 + dE and a*ke[i] >= np.pi:
                km.append(a * ke[i])
                kp.append(a * ke[Nsite - 1 - i])
                ckm.append(Vk[:-1, 0])
                ckp.append(Vk[:-1, 0])
                dkm[i] = Vk[-1,0]  
                dkp[Nsite - 1 - i] = Vk[-1,0]
    if pol == "upper":
        print("Finding coefficients for upper polariton")
        for i in range(Nsite//2, Nsite):
            Ek, Vk = Hlayer_v1(i, parameters, dat)
            #print("Elow = ", E0 - dE, "E = ", Ek[-1] , "E upper = ", E0 + dE)
            if Ek[-1]> E0-dE and Ek[-1]< E0 + dE and a*ke[i] >= np.pi:
                km.append(a * ke[i])
                kp.append(a * ke[Nsite - 1 - i])
                ckm.append(Vk[:-1,-1])
                ckp.append(Vk[:-1,-1])
                dkm[i] = Vk[-1,-1]
                dkp[Nsite - 1 - i] = Vk[-1,-1]
    tinit1 = time.time()
    ckp = np.array(ckp); ckm = np.array(ckm)
    print("first loop time = ", tinit1 - tinit0)

    print("kp size = ", len(kp))
    print("km size = ", len(km))
    kp = kp[::-1]


    tinit2 = time.time()
    kp0 = np.mean(kp)
    km0 = np.mean(km)
    αkp = np.exp(1.j*μg*np.array(kp)) * np.exp(-(kp-kp0)**2/(2*(σk**2)))
    αkm = np.exp(1.j*μg*np.array(km)) * np.exp(-(km-km0)**2/(2*(σk**2)))
    αkp_full = np.exp(1.j*μg*np.array(ke*a)) * np.exp(-(a*ke-kp0)**2/(2*(σk**2)))
    αkm_full = np.exp(1.j*μg*np.array(ke*a)) * np.exp(-(a*ke-km0)**2/(2*(σk**2)))

    print("Creating initial state")
    tinit2 = time.time()
    expnkp = np.exp(1.j*np.outer(range(Nsite), kp))/np.sqrt(Nsite)
    expnkm = np.exp(1.j*np.outer(range(Nsite), km))/np.sqrt(Nsite)
    ci = np.zeros(Nsite * (Nlayer + 1)) + 0.j
    for l in range(Nlayer):
        αkpbkp_repeat = np.tile(np.array(ckp)[:, l] * αkp, Nsite).reshape((Nsite, np.array(kp).shape[0])) * jp
        αkmbkm_repeat = np.tile(np.array(ckm)[:, l] * αkm, Nsite).reshape((Nsite, np.array(km).shape[0])) * jm
        ci[l*Nsite:(l+1)*Nsite] = ci[l*Nsite:(l+1)*Nsite] + np.sum(αkpbkp_repeat*expnkp, axis=1) * jp
        ci[l*Nsite:(l+1)*Nsite] = ci[l*Nsite:(l+1)*Nsite] + np.sum(αkmbkm_repeat*expnkm, axis=1) * jm
    ci[Nlayer*Nsite:(Nlayer+1)*Nsite] = ci[Nlayer*Nsite:(Nlayer+1)*Nsite] + dkp * αkp_full * jp
    ci[Nlayer*Nsite:(Nlayer+1)*Nsite] = ci[Nlayer*Nsite:(Nlayer+1)*Nsite] + dkm * αkm_full * jm
    ci = ci / np.sqrt(sum(abs(ci)**2))
    tinit3 = time.time()
    print("second loop time = ", tinit3 - tinit2)


    cmasked = np.zeros(Nlayer * Nsite + parameters.N_Pht) + 0.j
    cmasked[:Nlayer * Nsite] = ci[:Nlayer * Nsite]
    cmasked[Nlayer * Nsite:] = (ci[Nlayer * Nsite:])[mask]
    cmasked = cmasked / np.sqrt(sum(abs(cmasked)**2))
    print("cmasked normalization: ", sum(abs(cmasked[:])**2))
    plt.plot(abs(fft(cmasked[Nlayer * Nsite:], norm='ortho'))**2, 'r', label = "Photon, shape = " + str(cmasked.shape)+ ", Nlayer, Nsite, Npht = "+str(Nlayer)+', '+str(Nsite)+', '+str(parameters.N_Pht))
    plt.legend()
    plt.show()

    return cmasked
############################################################################################################
# Monte Carlo Method
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
    plt.suptitle('Exciton')
    plt.savefig(f'./InitialStates/Photonabswf_E0={param.E0 / param.eV:.2f}_dE={int(param.dE / param.eV * 1000)}.png')
    ax, fig = plt.subplots()
    plt.plot(abs(np.fft.ifft(Cik[1,:], norm='ortho'))**2, 'r')
    plt.suptitle('Phonon')
    plt.savefig(f'./InitialStates/Excitonabswf_E0={param.E0 / param.eV:.2f}_dE={int(param.dE / param.eV * 1000)}.png')
    return aFull,  c

def initElectronic(param, ntry = 100000, OPT = True, c = 0, a = 0):
    if OPT:
        a, c = initElectronicOPT(param, par.ntry)

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


# Generating the initial state of the electronic part:
def initstategen(parameters, dat, initState = 0):
    initype = parameters.initype
    Nsite = parameters.Nsite
    Nlayer = parameters.Nlayer
    if initype == 5:
        c = EW2KGauss(parameters, dat)
    if initype == 1:
        E0 = parameters.E0
        dE = parameters.dE
        eV = parameters.eV
        ntry_ = parameters.ntry
        print("--------------- \n Generator excits\n ---------------")
        data = np.load(f'./InitialStates/InitGentor_E0={E0 / eV:.2f}_dE={int(dE / eV * 1000)}.npy')
        cfull, _  = initElectronic(parameters, ntry = ntry_, OPT = False, a = data) 
        c = np.zeros(Nsite*Nlayer + parameters.N_Pht, dtype=complex)
        c[: Nsite * Nlayer] = (cfull[ : Nsite * Nlayer])
        c[Nsite * Nlayer:] = (cfull[Nsite * Nlayer:])[parameters.mask]
        #np.save(f'./InitialStates/InitState_Nlayer={Nlayer}_E0={E0 / eV:.2f}_dE={int(dE / eV * 1000)}.npy', c)
        fig, ax = plt.subplots()
        plt.plot(abs(c[: parameters.Nsite * parameters.Nlayer])**2)
        plt.savefig(f'./InitialStates/wfexciton_E0={E0 / eV:.2f}_dE={int(dE / eV * 1000)}.png')
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

# This function is used in split operator method to tiem evolve those k's with no coupling to the cavity
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


# This function is used in split operator method to shuffle the state in the block diagonal order
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

# This function is used in split operator method to shuffle BACK the state into the non-block diagonal order
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
            H2[0,1], H2[1,0] = np.sqrt(ω0/ωk[n]) * g * NormN,    np.sqrt(ω0/ωk[n]) * g * NormN
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
    kz = parameters.kz
    az = parameters.az
    Lz = parameters.Lz
    Rlz = parameters.Rlz
    mask = parameters.mask
    NormN = parameters.NormN
    Ndiag = (Nsite - Npht) * Nlayer

    ct_trot = ci + 0.j

    ct_trot[:Nsite*Nlayer] = np.exp(-1.j * 0.5 * dat.Hij * dt) * ct_trot[:Nsite*Nlayer]
    ct_trot[Nsite*Nlayer:] = np.exp(-1.j * 0.5 * dat.HLoss * dt) * ct_trot[Nsite*Nlayer:]
    #ct_trot = np.exp(-1.j * 0.5 * dt * dat.Hij) * ct_trot
    
    tfft0 = time.time()
    ct_trot2D = np.reshape(ct_trot[:Nsite*Nlayer], (Nlayer, Nsite)).T
    ct_trot[:Nsite*Nlayer]= (fft(ct_trot2D, axis=0, norm = "ortho").T).flatten()
    # for nl in range(Nlayer):
    #     ct_trot[Nsite*nl:Nsite*(nl+1)] = fft(ct_trot[Nsite*nl:Nsite*(nl+1)], norm = 'ortho')
    tfft1 = time.time()

    tsh0 = time.time()
    ct_shuf = Ushuf(parameters, ct_trot)
    tsh1 = time.time()

    ct_shuf = np.exp(-1.j * dt * Enoncouple) * ct_shuf

    tev0 =  time.time()
    ##----------------------------------------------
    # converting the 1D array to 2D array of each block
    cres1 = np.reshape(ct_shuf[Ndiag : Ndiag + (Npht)*(Nlayer+1)], (Npht, Nlayer+1))
    cres2 = np.zeros((Npht, 2), dtype=complex)
    ###----------------- tlc2 in splitEvolution
    cres2[:, 0] =  cres1[:,:-1] @ (np.sin(kz*((Rlz * az) + Lz/2))/NormN)
    cres2[:, 1] = cres1[:,-1]
    ###----------------- tlc2p in splitEvolution
    cres3 = np.matmul(dat.V2_list, cres2.reshape((Npht, 2, 1))).reshape((Npht, 2))
    ###----------------- tlc2p_E in splitEvolution
    cres4 = np.exp(-1.j * np.array(dat.E2_list) * dt) * cres3
    ###----------------- tlc2p_p in splitEvolution
    cres5 = np.matmul(np.transpose(np.conjugate(dat.V2_list), axes=(0, 2, 1)), cres4.reshape((Npht, 2, 1))).reshape((Npht, 2))
    ###----------------- Δc in splitEvolution
    Δc = cres5[:,0].T - cres2[:,0] * np.exp(-1.j * εk[mask] * dt)
    cdtres = np.zeros((Npht, Nlayer + 1), dtype=complex)
    ###----------------- first cdt[:-1] in splitEvolution
    cdtres[:, :-1] = cdtres[:,:-1] + (cres1[:, :-1].T * np.exp(-1.j * εk[mask] * dt)).T
    ###----------------- second cdt[:-1] in splitEvolution
    cdtres[:, :-1] = cdtres[:, :-1] + np.outer(Δc , np.sin(kz*((Rlz * az) + Lz/2))/NormN)
    ###----------------- second cdt[:-1] in splitEvolution
    cdtres[:, -1] = cres5[:, 1]
    ct_shuf[Ndiag : Ndiag + (Npht)*(Nlayer+1)] = cdtres.flatten()
    ##----------------------------------------------
    tev1 =  time.time()
    #print("split op Evolution = ", (tev1-tev0))

    tsh2 = time.time()
    ct_trot = Ushufrev(parameters, ct_shuf)
    tsh3 = time.time()
    #print("Shufling = ", (tsh1-tsh0) )
    #print("ReShufling = ", (tsh3-tsh2))

    tfft2 = time.time()
    ict_trot2D = np.reshape(ct_trot[:Nsite*Nlayer], (Nlayer, Nsite)).T
    ct_trot[:Nsite*Nlayer]= (ifft(ict_trot2D, axis=0, norm = "ortho").T).flatten()
    # for nl in range(Nlayer):
    #     ct_trot[Nsite*nl:Nsite*(nl+1)] = ifft(ct_trot[Nsite*nl:Nsite*(nl+1)], norm = 'ortho')
    tfft3 = time.time()
    #print("Fourier transform = ", (tfft1-tfft0) + (tfft3-tfft2))

    ct_trot[Nsite*Nlayer:] = np.exp(-1.j * 0.5 * dat.HLoss * dt) * ct_trot[Nsite*Nlayer:]
    ct_trot[:Nsite*Nlayer] = np.exp(-1.j * 0.5 * dat.Hij * dt) * ct_trot[:Nsite*Nlayer]
    #ct_trot = np.exp(-1.j * 0.5 * dt * dat.Hij) * ct_trot
    #print(">>>>> Propagate time = ", time.time() - tp0)
    return ct_trot


# To calculate the wavefront of each trajectory
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

# finding Mean Square Displacement (MSD)
def msd(Nsite, Nlayer, mydata):
    r = np.tile(np.arange(Nsite) - Nsite//2, Nlayer)
    x2 = np.zeros(mydata.shape[1])
    for i in range(mydata.shape[1]):
        pn = mydata[:Nsite*Nlayer, i]
        x2[i] = np.sum(r**2 * pn) - np.sum(r * pn)**2
    return x2

#@jit(nopython=False)
def Force(γ, dH0, ci):
    F = -dH0 #np.zeros((len(dat.R)))
    Nex = dH0.shape[0]
    F -= γ * ((ci[:Nex]).conj() * ci[:Nex]).real
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
    #ci /= np.sum(ci.conjugate()*ci) 
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
    F2 = Force(par.γ, dat.dH0, dat.ci) #Force(par.γ, dat.dH0, dat.ci) # force at t2
    v += 0.5 * (F1 + F2) * par.dtN / par.M
    dat.F1 = F2
    dat.P = v * par.M
    # ======================================================
    # half electronic evolution
    for t in range(int(np.ceil(EStep/2))):
        ci = propagateCi(par, dat, ci, dtE) 
    #ci /= np.sum(ci.conjugate()*ci)  
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
    rho_ensemble = np.zeros((1 + Nsite * Nlayer + Npht, NSteps//nskip + pl, NTraj), dtype=complex)
    psi_t =        np.zeros((Nsite * Nlayer + Npht, NSteps//nskip + pl, NTraj), dtype=complex)
    Layersaddup =  np.zeros((Nsite + Npht, NSteps//nskip + pl, NTraj), dtype=complex)
    wavefront =    np.zeros((NSteps//nskip + pl, NTraj), dtype=complex)
    MSD_exc =          np.zeros((NSteps//nskip + pl, NTraj))
    # Ensemble
    #dat.Hij0 = parameters.Hel0()
    #dat.E0, dat.V0 = np.linalg.eigh(dat.Hij0)
    dat.Enoncouple = Enoncouplefunc(parameters)
    dat.E2_list, dat.V2_list = BlocksED(parameters)
    dat.Pht_ks = np.arange(Nsite)[mask]
    tinit0 = time.time()
    # Call function to initialize mapping variables
    dat.ct0 = initstategen(parameters, dat, initState = 0) # np.array([0,1])
    dat.HLoss = parameters.HelL()
    print("Initial state:", time.time() - tinit0)
    for itraj in range(NTraj):
        seednum = ***jobnum*** + itraj
        # Trajectory data
        dat.R, dat.P = parameters.initR(seednum)
        
        # set propagator
        vv  = VelVer
        dat.ci = dat.ct0[:] * 1.0
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
                #Eikn = np.exp(-1.j * np.outer(Rn, kph * a))/np.sqrt(Nsite)
                rho_ensemble[0,iskip, itraj] = i
                cph = fft(dat.ci[Nsite * Nlayer:], norm = 'ortho')
                rho_ensemble[1:Nsite * Nlayer+1,iskip, itraj] += np.abs(dat.ci[:Nsite * Nlayer])**2 
                rho_ensemble[Nsite * Nlayer+1:,iskip, itraj] += np.abs(cph)**2 
                psi_t[:Nsite * Nlayer,iskip, itraj] += dat.ci[:Nsite * Nlayer]
                psi_t[Nsite * Nlayer:,iskip, itraj] += cph
                iskip += 1
            #-------------------------------------------------------
            dat = vv(dat)
        time_taken = time.time()-t0
        print(f"Time taken: {time_taken} seconds")
        for nlayer in range(Nlayer):
            Layersaddup[:Nsite, :, itraj] += rho_ensemble[nlayer * Nsite + 1:(nlayer+1)*Nsite + 1, :, itraj]
        Layersaddup[Nsite:, :, itraj] += rho_ensemble[Nlayer * Nsite + 1:, :, itraj]
        print("finding group velocity")
        wavefront[:, itraj] = wavefront_x_vs_t(Nsite, 1, Layersaddup[:Nsite, :, itraj].T)
        print("Group Velocity found")
        MSD_exc[:, itraj] = msd(Nsite, Nlayer, rho_ensemble[1:Nsite * Nlayer + 1, :, itraj])
        print("MSD found")

    return rho_ensemble, psi_t, Layersaddup, wavefront, MSD_exc