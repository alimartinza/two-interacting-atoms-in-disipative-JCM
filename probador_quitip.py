from qutip import *
import numpy as np
import pandas as pd

psi0=tensor(basis(2,0),basis(2,1))
rho0=ket2dm(psi0)


e=basis(2,0)
gr=basis(2,1)

ee0=tensor(e,e,basis(3,0))
ee1=tensor(e,e,basis(3,1))
ee2=tensor(e,e,basis(3,2))

eg0=tensor(e,gr,basis(3,0))
ge0=tensor(gr,e,basis(3,0))

eg1=tensor(e,gr,basis(3,1))
ge1=tensor(gr,e,basis(3,1))

eg2=tensor(e,gr,basis(3,2))
ge2=tensor(gr,e,basis(3,2))

gg0=tensor(gr,gr,basis(3,0))
gg1=tensor(gr,gr,basis(3,1))
gg2=tensor(gr,gr,basis(3,2))

base=[gg0,gg1,(eg0+ge0).unit(),(eg0-ge0).unit(),gg2,(eg1+ge1).unit(),ee0,(eg1-ge1).unit(),(eg2+ge2).unit(),ee1,(eg2-ge2).unit(),ee2]

n=tensor(qeye(2),qeye(2),num(3))
a=tensor(qeye(2),qeye(2),destroy(3))

# print(eg1+2*eg2+3*ge0+4*ge1+5*ge2+6*gg0+7*gg1+8*gg2)

rho=[ket2dm(eg1),ket2dm(gg2),ket2dm((eg0+ge0).unit())]
eigenenergievector=np.empty([len(rho),12])
eigenenstatesvetor=np.empty([len(rho),12,12,1],dtype="complex")
data=pd.DataFrame()
data['expect']=np.zeros(len(rho))
# print(data['EigenE'][0])
for j,d_matrix in enumerate(rho):
    eigenvals,eigenvecs=d_matrix.eigenstates()
    for i,vec in enumerate(eigenvecs):
        eigenenstatesvetor[j][i]=vec.full()
data['EigenEnergies']=eigenenergievector
data['EigenStates']=eigenenstatesvetor



# for vec in eigenvecs:
#     print(vec)
# print(type(eigenvecs[0]))
    #vals = np.array(rho[i].eigenenergies())