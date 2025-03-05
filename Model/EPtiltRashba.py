import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

# Model Hamiltonian 
# H = ∑_k βx_k |k⟩⟨k| - τ ∑_k(|k⟩⟨k+1|+|k+1⟩⟨k|) + 1/2 ∑_k (mv^2_k + Kx^2_k) 

class parameters():
    initype = 5 #EW2KGauss
    amu, ps, cm, Å =  1836.0, 41341.37, 1/0.000004556, 1.8897259885789
    c, π = 137.036, np.pi
    eV = 1/27.2114
    Nsite = ***nsites***
    Nlayer = ***Nlayer***
    initState = Nsite//2
    NSteps = int(.2*420) # Number of time steps for dtN
    NTraj = 1 # number of initial state sampling
    dtN = 150 # Nucleat time steps
    dtE = dtN/***dtE*** # electronic time steps
    M = 1.0 # nuclear mass
    nskip = 5 # ???
    τ = 150/cm # Hopping integral contsant for cleane system
    β = 2*1052.8# a.u. is 300K
    K = 14500.0 * (amu/ps**2) # Phonons natural mode K = m * ω^2
    ms = 250 * amu 
    γ = 0*3500/(cm * Å)/ms**0.5
    a = 1000 * Å
    az = 105 * Å
    ω = (K/ms) ** 0.5 # Phonon frequency
    g = 2 * 1500/cm
    ϵ0 = 3 * eV
    mz = 1
    ################
    Lz = 4500* Å #0.01 * Nsite * a * Å
    kz = (np.pi/Lz)
    if Nsite%2==0:
        Rn = np.arange(-Nsite//2,Nsite//2) # sites position
        ke =  2 * π * (Rn)/(Nsite*a) # Kpoints
    if Nsite%2==1:
        Rn = np.arange(-(Nsite-1)//2, (Nsite+1)//2)
        ke =  2 * π * (Rn)/(Nsite*a)
    if Nlayer%2==0:
        Rlz = np.arange(-Nlayer//2,  Nlayer//2) # sites position
    if Nlayer%2==1:
        Rlz = np.arange(-(Nlayer-1)//2, (Nlayer+1)//2)
    ω0 = c*kz
    ωk = c*np.sqrt((ke)**2 + (kz)**2)
    ## Finding Phonon cut-off
    εk = ϵ0*np.ones(Nsite)-2*τ*np.cos(ke*a)
    diff = (2*(g**2)*ωk/ω0)/(2*(ωk-εk)**2)
    diff_tresh = 0.05
    diff[:]=10*diff_tresh
    #diff[int((diff.shape[0]//2)*0.95):-int((diff.shape[0]//2)*0.95)] = 20+20*max(abs(diff))
    mask = diff >= diff_tresh
    N_Pht = ωk[mask].shape[0]
    NStates = Nsite + N_Pht # computational basis size
    Δm = ***Δm***  # Integer defining the k-point Δk ∝ Δm
    sinθ = Δm * 2 * π / (kz*Nsite*a) 
    θ = np.arcsin(Δm * 2 * π / (kz*Nsite*a)) # angle of tilte to match with Δm
    NormN = 1
    Emin = 0.5 * ((ϵ0 - 2*τ + ω0) - np.sqrt((ϵ0 - 2*τ - ω0)**2 + 4*((g*NormN)**2))) # Minimum energy assuming no tilte and disorder
    # plt.plot(ke[mask], ωk[mask], label="photon")
    # plt.plot(ke, diff, label="diff")
    # plt.plot(ke, εk, label = "exciton")
    # plt.legend()
    # plt.show()
    ################
    print("arg arcsin:", Δm * 2 * π /(kz*Nsite*a))
    print("θ=", np.round(θ,3), " radian, ", np.round(θ*180/np.pi,3), " degree")
    print("N site = ", Nsite)
    print("Nlayer = ???")
    print("N Photon = ", N_Pht)
    print("NStates = ", NStates)
    print("Emin = ", np.round(Emin, 5))
    print("β = ", β)
    print("τ = ", np.round(τ, 5))
    print("ϵ0 = ", np.round(ϵ0,5))
    print("g = ", np.round(g, 5))
    print("γ = ", np.round(γ,6))
    print("ω (phonon) = ", np.round(ω, 5))
    print("ω0 = ", np.round(ω0,5))
    print("kz = ", np.round(kz,5))
    print("mz = ???")
    print("Lz = ", np.round(Lz,2), ", a = ", np.round(a, 2), ", N*a = ", np.round(Nsite*a,2))
    print("NSteps = ", NSteps)
    print("NTraj = ", NTraj)
    print("dtN = ", np.round(dtN,4), ", dtE = ", np.round(dtE,4))


def Hel0():
    τ = parameters.τ
    N = parameters.Nsite
    Npht = parameters.N_Pht
    sinθ = parameters.sinθ
    a = parameters.a
    γ  = parameters.γ
    g  = parameters.g
    ϵ0 = parameters.ϵ0
    kz = parameters.kz
    ωk = parameters.ωk
    mask = parameters.mask
    ω0 = parameters.ω0
    ke = parameters.ke
    Rn = parameters.Rn
    
    mymat = np.zeros((N + Npht, N + Npht), dtype=complex)
    Vhop = -τ*(np.eye(N, k = +1)+np.eye(N, k = -1))
    Vhop[-1, 0], Vhop[0, -1] =  -τ, -τ
    Vonsite = np.diag( ϵ0 * np.ones(N))
    mymat[:N,:N] = mymat[:N, :N] + Vhop[:N, :N] + Vonsite[:N, :N]
    
    Rz = Rn * a * sinθ
 
    repeated_ωω = np.tile(np.sqrt(ωk/ω0), N).reshape((N, N))
    coskz = (np.tile(np.cos(kz*Rz), N).reshape((N, N))).T
    mymat[:N,N:] = (g * repeated_ωω * coskz * np.exp(1.j*np.outer(Rn,ke*a))/np.sqrt(N))[:, mask]
    mymat[N:,:N] = np.conjugate(mymat[:N,N:].T)
    
    mymat[N:,N:] = np.diag(ωk[mask])
    return mymat


def HelR(R):
    N = parameters.Nsite
    Npht = parameters.N_Pht
    γ  = parameters.γ
    kz = parameters.kz
    Rn = parameters.Rn
    
    mymat = np.zeros((N + Npht, N + Npht), dtype=complex)
    Vonsite = np.diag(R * γ )
    mymat[:N,:N] = Vonsite[:N, :N]
    
    return np.diag(mymat)


def dHel0(R):
    ω = parameters.ω
    return  (ω**2) * R

# def dHel(R):
#     N = parameters.NStates
#     γ = parameters.γ

#     dHijk = np.zeros((N, N, N ))
#     #dHijk = np.einsum("k -> kkk")
#     for k in range(N):
#         dHijk[k,k,k] = γ
#     return dHijk


def initR():
    R0 = 0.0
    P0 = 0.0
    β  = parameters.β
    ω  = parameters.ω
    K = parameters.K
    M = parameters.M
    N = parameters.Nsite
    Npht = parameters.N_Pht

    sigP = np.sqrt( ω / ( 2 * np.tanh( 0.5*β*ω ) ) )
    sigR = sigP/ω

    R = np.zeros(N+Npht)
    P = np.zeros(N+Npht)
    for d in range(N):
        R[d] = np.random.normal()*sigR
        P[d] = np.random.normal()*sigP  
    # print(np.sqrt(β/K))
    # print(np.sqrt(β/M))
    # R = np.random.normal(scale=np.sqrt(β/K),size=(ndof))
    # P = np.random.normal(scale=np.sqrt(β/M),size=(ndof))
    return R, P