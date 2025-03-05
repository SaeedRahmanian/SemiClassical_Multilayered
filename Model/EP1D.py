import numpy as np
import scipy as sp
import random
import matplotlib.pylab as plt
from scipy.sparse import csr_matrix

# Model Hamiltonian 
# H = ∑_k βx_k |k⟩⟨k| - τ ∑_k(|k⟩⟨k+1|+|k+1⟩⟨k|) + 1/2 ∑_k (mv^2_k + Kx^2_k) 

class parameters():
    initype = 1 #0 = singlestate, 1 = Montecarlo, 5 = EW2KGauss
    ntry = 300000
    amu, ps, cm, Å =  1836.0, 41341.37, 1/0.000004556, 1.8897259885789
    c, π = 137.036, np.pi
    c = c/2.4
    eV = 1/27.2114
    Photoncut = 'cut'
    E0 = ***E0*** * eV
    dE = ***dE*** * eV
    branch = 0
    Nsite = 40001
    Nlayer = ***Nlayer***
    initState = Nsite//2
    NSteps = ***NSteps***#int(0.05*420) # Number of time steps for dtN
    NTraj = ***NTraj*** # number of initial state sampling
    dtN = 20 # Nucleat time step1
    dtE = dtN/40 # electronic time steps
    nskip = 15 # ???
    M = 1.0 # nuclear mass
    τ = 0 * 100/cm # Hopping integral contsant for cleane system
    β = 1052.8# a.u. is 300K is 1052.8
    γ = ***γ*** * 1.1 * ((1440/cm)**1.5) 
    Γ = 0.0
    a = 12 * Å
    az = 40 * Å
    ω = 1440/cm 
    ϵ0 = 3.2 * eV
    mz = 1
    if Nlayer%2==0:
        Rlz = np.arange(-Nlayer//2,  Nlayer//2) # sites position
    if Nlayer%2==1:
        Rlz = np.arange(-(Nlayer-1)//2, (Nlayer+1)//2)
    ######## Cavity and Fourier parameters
    Lz = 1000 * Å # 0.01 * Nsite * a * Å
    kz = mz*(np.pi/Lz)
    NormN = np.sqrt(sum(np.sin(kz * ((Rlz * az) + Lz/2))**2))
    g = (1/NormN) * 2.6 * 1500/cm
    # if Nsite%2==0:
    #     Rn = np.arange(-Nsite//2,Nsite//2) # sites position
    #     ke =  2 * π * (Rn)/(Nsite*a) # Kpoints
    # if Nsite%2==1:
    #     Rn = np.arange(-(Nsite-1)//2, (Nsite+1)//2)
    #     ke =  2 * π * (Rn)/(Nsite*a)
    Rn = np.arange(0, Nsite)
    ke =  2 * π * (Rn)/(Nsite*a) # Kpoints
    ω0 = c*kz
    ωk = np.roll( c * np.sqrt((ke - ke[Nsite//2])**2 + (kz)**2), -1 * (Nsite//2))
    ##### Finding Phonon cut-off
    εk = ϵ0*np.ones(Nsite)-2*τ*np.cos(ke*a) # Electronic band energies
    diff = (2*((g*NormN)**2)*ω0/ωk)/(2*(ωk-εk)**2) 
    diff_tresh, diff_max = 0.005, max(abs(diff))
    if Photoncut == "full":
        diff[:]=10*diff_tresh + 10
    if Photoncut == "cut":
        diff[:int(Nsite*0.10)] = 20 * diff_max + 10
        diff[-int(Nsite*0.10):] = 20 * diff_max + 10
    mask = diff >= diff_tresh
    # if g==0:
    #     mask = mask*False
    N_Pht = ωk[mask].shape[0] # Photon number
    NStates = Nlayer * Nsite + N_Pht # computational basis size
    Δm = 0  # Integer defining the k-point Δk ∝ Δm
    if Nlayer > 1:
        Δm = 0
    θ = np.arcsin(Δm * 2 * π / (kz*Nsite*a)) # angle of tilte to match with Δm (sinθ = Δm * 2 * π / (kz*Nsite*a) )
    sinθ = np.sin(θ)
    Emin = 0.5 * ((ϵ0 - 2*τ + ω0) - np.sqrt((ϵ0 - 2*τ - ω0)**2 + 4*((g*NormN)**2))) # Minimum energy assuming no tilte and disorder
    ####### Plotting lsyer in cavity
    plt.plot((Rn - Rn[Nsite//2])*a, Rn*0 + (Lz), 'b', lw = 4)
    plt.plot((Rn - Rn[Nsite//2])*a, Rn*0 , 'b', lw = 4)
    for nLay in range(Nlayer):
        plt.plot((Rn - Rn[Nsite//2])*a*np.cos(θ), (Rlz[nLay] * az) + ((Rn - (Nsite//2)) * a * sinθ) + Lz/2, 'k')
    plt.plot((1+Nsite//4)*a*np.sin(kz*np.arange(0, Lz, 1)), np.arange(0, Lz, 1), "r")
    plt.plot((Rn - Rn[Nsite//2])*1.2*a, Rn*0 + (Lz/2), '--b', lw = 2)
    #plt.savefig("Systemsetup.png")
    ### Plotting mask
    # plt.plot(ke[mask], ωk[mask], label="photon")
    # plt.plot(ke, diff, 'o', label="diff")
    # plt.plot(ke, εk, label = "exciton")
    # plt.legend()
    # plt.show()
    ################
    print("c = ", np.round(c,5))
    print("inital branch = ", ['LP', 'UP'][branch])
    print("E0 = ", E0/eV, "eV, dE = ", dE /eV, 'eV')
    print("arg arcsin:", Δm * 2 * π /(kz*Nsite*a))
    print("θ=", np.round(θ,3), " radian, ", np.round(θ*180/np.pi,3), " degree")
    print("N site = ", Nsite)
    print("Nlayer = ", Nlayer)
    print("N Photon = ", N_Pht)
    print("NStates = ", NStates)
    print("Emin = ", np.round(Emin, 5)/eV, "eV")
    print("β = ", β)
    print("τ = ", np.round(τ, 5)/eV, "eV")
    print("ϵ0 = ", np.round(ϵ0,5)/eV, "eV")
    print("g = ", np.round(g*NormN, 5)/eV, "/NormN eV")
    print("γ = ", np.round(γ,6)/eV, "eV")
    print("ω (phonon) = ", np.round(ω, 5)/eV, "eV")
    print("ω0 = ", np.round(ω0,5)/eV, "eV")
    print("kz = ", np.round(kz,5))
    print("mz = ", mz)
    print("Lz = ", np.round(Lz,2), ", az = ", np.round(az, 2), ", a = ", np.round(a, 2), ", N*a = ", np.round(Nsite*a,2))
    print("NSteps = ", NSteps)
    print("NTraj = ", NTraj)
    print("dtN = ", np.round(dtN,4), ", dtE = ", np.round(dtE,4))



def Hel0():
    τ = parameters.τ
    N = parameters.Nsite
    Nlayer = parameters.Nlayer
    Npht = parameters.N_Pht
    sinθ = parameters.sinθ
    a = parameters.a
    az = parameters.az
    g  = parameters.g
    ϵ0 = parameters.ϵ0
    kz = parameters.kz
    ωk = parameters.ωk
    mask = parameters.mask
    Lz = parameters.Lz
    ω0 = parameters.ω0
    ke = parameters.ke
    Rn = parameters.Rn
    Rlz = parameters.Rlz
    
    mymat = np.zeros((Nlayer * N + Npht, Nlayer * N + Npht), dtype=complex)
    Vhop = -τ*(np.eye(N, k = +1)+np.eye(N, k = -1))
    Vhop[-1, 0], Vhop[0, -1] =  -τ, -τ
    for iLay in range(Nlayer):
        mymat[N * iLay: N * iLay + N, N * iLay: N * iLay + N] = Vhop
    Vonsite = ϵ0 * np.eye(Nlayer * N)
    mymat[:N*Nlayer, :N*Nlayer] = mymat[:N*Nlayer, :N*Nlayer] + Vonsite
    
    repeated_ωω = np.tile(np.sqrt(ω0/ωk), N).reshape((N, N))
    for nLay in range(Nlayer):
        Rz = (Rlz[nLay] * az) + Lz/2 + (Rn * a * sinθ) - ((N//2) * a * sinθ) 
        sinkz = (np.tile(np.sin(kz*Rz), N).reshape((N, N))).T
        mymat[N*nLay: N*nLay + N, N * Nlayer:] = (g * repeated_ωω * sinkz * np.exp(1.j*np.outer(Rn,ke*a))/np.sqrt(N))[:, mask]
        mymat[N*Nlayer:, N*nLay: N*nLay + N] = np.conjugate(mymat[N*nLay: N*nLay + N, N * Nlayer:].T)
    
    mymat[N * Nlayer:, N * Nlayer:] = np.diag(ωk[mask])
    
    return mymat


def HelR(Rdis):
    γ  = parameters.γ
    return Rdis * γ


def HelL():
    Nsite = parameters.Nsite
    Nlayer = parameters.Nlayer
    Npht = parameters.N_Pht
    Γ  = parameters.Γ
    Ar = np.ones(Npht) + 0.j
    return -1.j * Γ * 0.5 * Ar


def dHel0(R):
    ω = parameters.ω
    return  (ω**2) * R


def initR(seednum):
    np.random.seed(seednum)
    R0 = 0.0
    P0 = 0.0
    β  = parameters.β
    ω  = parameters.ω
    M = parameters.M
    N = parameters.Nsite
    Nlayer = parameters.Nlayer
    Npht = parameters.N_Pht

    sigP = np.sqrt( ω / ( 2 * np.tanh( 0.5*β*ω ) ) )
    sigR = sigP/ω

    Rd = np.zeros((Nlayer * N))
    Pd = np.zeros((Nlayer * N))
    for d in range(Nlayer * N):
        Rd[d] = np.random.normal()*sigR
        Pd[d] = np.random.normal()*sigP
    #np.save("Rd"+str(seednum)+".npy", Rd)
    print("Rd: ", np.round(Rd[:20],3))
    return Rd, Pd
