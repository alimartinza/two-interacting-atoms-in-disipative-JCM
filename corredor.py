# from qutip import *
# import numpy as np
# import matplotlib.pyplot as plt
# import jcm_lib as jcm
# from matplotlib import cm
# import os

# script_path= os.path.dirname(__file__)


# #DEFINIMOS LOS OPERADORES QUE VAMOS A USAR EN LOS CALCULOS
# n=tensor(qeye(2),qeye(2),num(3))
# sqrtN=tensor(qeye(2),qeye(2),Qobj(np.diag([0,1,np.sqrt(2)])))
# n2=tensor(qeye(2),qeye(2),Qobj(np.diag([0,1,4])))
# a=tensor(qeye(2),qeye(2),destroy(3))
# sm1=tensor(sigmam(),qeye(2),qeye(3))
# sp1=tensor(sigmap(),qeye(2),qeye(3))
# sz1=tensor(sigmaz(),qeye(2),qeye(3))
# sx1=tensor(sigmax(),qeye(2),qeye(3))
# sm2=tensor(qeye(2),sigmam(),qeye(3))
# sp2=tensor(qeye(2),sigmap(),qeye(3))
# sz2=tensor(qeye(2),sigmaz(),qeye(3))
# sx2=tensor(qeye(2),sigmax(),qeye(3))

# #DEFINIMOS LOS VECTORES DE LA BASE
# e=basis(2,0)
# gr=basis(2,1)

# e0=tensor(e,basis(3,0))
# g0=tensor(gr,basis(3,0))
# g1=tensor(gr,basis(3,1))
# sx=tensor(sigmax(),qeye(3))
# sy=tensor(sigmay(),qeye(3))
# sz=tensor(sigmaz(),qeye(3))
# sp=tensor(sigmap(),qeye(3))
# sm=tensor(sigmam(),qeye(3))


# ee0=tensor(e,e,basis(3,0)) #0
# ee1=tensor(e,e,basis(3,1)) #1
# ee2=tensor(e,e,basis(3,2)) #2

# eg0=tensor(e,gr,basis(3,0)) #3
# ge0=tensor(gr,e,basis(3,0)) #6

# eg1=tensor(e,gr,basis(3,1)) #4
# ge1=tensor(gr,e,basis(3,1)) #7

# eg2=tensor(e,gr,basis(3,2)) #5
# ge2=tensor(gr,e,basis(3,2)) #8

# gg0=tensor(gr,gr,basis(3,0)) #9
# gg1=tensor(gr,gr,basis(3,1)) #10
# gg2=tensor(gr,gr,basis(3,2)) #11



# SMALL_SIZE = 15
# MEDIUM_SIZE = 15
# BIGGER_SIZE = 20

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# plt.rc('figure.subplot',left=0.1)
# plt.rc('figure.subplot',bottom=0.102)
# plt.rc('figure.subplot',right=0.962)
# plt.rc('figure.subplot',top=0.95)


# N_c=5

# M=np.zeros((4*N_c,4*N_c))
# M[0,3*N_c]=1
# M[1,3*N_c+1]=1
# M[2,N_c]=1/np.sqrt(2)
# M[2,2*N_c]=1/np.sqrt(2)
# M[3,N_c]=1/np.sqrt(2)
# M[3,2*N_c]=-1/np.sqrt(2)

# for ii in range(1,N_c-1):
#     M[4*ii,3*N_c+1+ii]=1
# for ii in range(1,N_c-1):
#     M[4*ii+1,N_c+ii]=1/np.sqrt(2)
#     M[4*ii+1,2*N_c+ii]=1/np.sqrt(2)
#     M[4*ii+3,N_c+ii]=1/np.sqrt(2)
#     M[4*ii+3,2*N_c+ii]=-1/np.sqrt(2)
# for ii in range(1,N_c-1):
#     M[4*ii+2,ii-1]=1

# n=tensor(qeye(2),qeye(2),num(N_c)).transform(M)
# # sqrtN=tensor(qeye(2),qeye(2),Qobj(np.diag([0,1,np.sqrt(2)])))
# n2=tensor(qeye(2),qeye(2),Qobj(np.diag([i*i for i in range(N_c)]))).transform(M)
# a=tensor(qeye(2),qeye(2),destroy(N_c)).transform(M)
# sm1=tensor(sigmam(),qeye(2),qeye(N_c)).transform(M)
# sp1=tensor(sigmap(),qeye(2),qeye(N_c)).transform(M)
# sz1=tensor(sigmaz(),qeye(2),qeye(N_c)).transform(M)
# sx1=tensor(sigmax(),qeye(2),qeye(N_c)).transform(M)
# sm2=tensor(qeye(2),sigmam(),qeye(N_c)).transform(M)
# sp2=tensor(qeye(2),sigmap(),qeye(N_c)).transform(M)
# sz2=tensor(qeye(2),sigmaz(),qeye(N_c)).transform(M)
# sx2=tensor(qeye(2),sigmax(),qeye(N_c)).transform(M)

# n_old=tensor(qeye(2),qeye(2),num(N_c))
# # sqrtN=tensor(qeye(2),qeye(2),Qobj(np.diag([0,1,np.sqrt(2)])))
# n2_old=tensor(qeye(2),qeye(2),Qobj(np.diag([i*i for i in range(N_c)])))
# a_old=tensor(qeye(2),qeye(2),destroy(N_c))
# sm1_old=tensor(sigmam(),qeye(2),qeye(N_c))
# sp1_old=tensor(sigmap(),qeye(2),qeye(N_c))
# sz1_old=tensor(sigmaz(),qeye(2),qeye(N_c))
# sx1_old=tensor(sigmax(),qeye(2),qeye(N_c))
# sm2_old=tensor(qeye(2),sigmam(),qeye(N_c))
# sp2_old=tensor(qeye(2),sigmap(),qeye(N_c))
# sz2_old=tensor(qeye(2),sigmaz(),qeye(N_c))
# sx2_old=tensor(qeye(2),sigmax(),qeye(N_c))

# # M[-1]=np.zeros(4*N_c) #Esta columna deberia pertenecer al gg,n+1, pero no existe asi que la matriz tiene 0's en esta fila. Para poder invertirla le ponemos un 1 en el estado een, para que el een se mapee al een, y listo. El estado gg,N+1 y gg,N+1 no estan disponibles
# # M[-2]=np.zeros(4*N_c)
# # M[-3]=np.zeros(4*N_c)



# # sz1=tensor(sigmaz(),qeye(2),qeye(N_c))
# # sz1_new=sz1.transform(M)
# # sz1_new.tidyup(atol=1e-10)

# with open("output.txt", "a") as file_object:
#     print("-------------------------------------------------------------------------------------------------------------------------------------------", file=file_object)
#     print(f"TERMINAL {script_path} corredor.py", file=file_object)
#     print("M", file=file_object)
#     print(M, file=file_object)  
#     # print("M@M.T.conj()", file=file_object)
#     # A=M@M.T.conj()
#     # A[np.abs(A)<=1e-8] = 0
#     # print(A, file=file_object)
#     # print("a old", file=file_object)
#     # print(a_old, file=file_object) 
#     # print("a", file=file_object)
#     # a.tidyup()
#     # print(a, file=file_object) 

import numpy as np
from numpy.linalg import eigh

rho=np.array([[1,1],[1,1]])
def rankV_func(rho):
    eigvals, eigvecs = eigh(rho) #primero buscamos los autovalores y autovectores de la matriz dada por dos razones
    rank = len(eigvals)
    V = np.dot(eigvecs, np.power(np.diag(eigvals),1/2)) #la segunda y mas fundamental es para encontrar esta matriz que vendria a ser la descomposicion espectral de rho: rho=V^{\dagger}V. Esta matriz V nos sirve para ir transformandola con la matriz unitaria arbitraria e ir buscando el minimo de entropia
    return rank, V,eigvals,eigvecs
rank,V,eigvals,eigvecs=rankV_func(rho)
print('rank')
print(rank)
print('eigvals')
print(eigvals)
print('shape eigenvecs')
print(eigvecs.shape)
print('eigvecs')
print(eigvecs)
print('V')
print(V)

