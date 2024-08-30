from qutip import *
import numpy as np

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

n=tensor(qeye(2),qeye(2),num(3))
a=tensor(qeye(2),qeye(2),destroy(3))

# print(eg1+2*eg2+3*ge0+4*ge1+5*ge2+6*gg0+7*gg1+8*gg2)
def entropy_vn(rho):
        """
        Von-Neumann entropy of density matrix

        Parameters
        ----------
        rho : qobj or list of qobjs
            Density matrix.
        base : {e,2}
            Base of logarithm.
        sparse : {False,True}
            Use sparse eigensolver.

        Returns
        -------
        entropy : list of floats
            Von-Neumann entropy of `rho`.

        Examples
        --------
        >>> rho=0.5*fock_dm(2,0)+0.5*fock_dm(2,1)
        >>> entropy_vn(rho,2)
        1.0

        """

        s=np.zeros(len(rho))
        eigenvals=[]
        eigenvecs=[]
        print(eigenvecs)
        for i in range(len(rho)):

            if rho[i].type == 'ket' or rho[i].type == 'bra':
                rho[i] = ket2dm(rho[i])
            rho[i]=rho[i].tidyup()
            eigenvals[i],eigenvecs[i] = rho[i].eigenstates()
            eigenvecs[i]=np.array(eigenvecs[i].full())
            nzvals = eigenvals[eigenvals > 0]
            s[i] = float(np.real(-sum(nzvals * np.log(nzvals))))

        return s,eigenvals,eigenvecs

dm=ket2dm(gg0)
eigenen,eigenvecs=dm.eigenstates()
print(eigenen)
print(type(eigenen))


# entropy_vn(dm)

vals=ket2dm(gg0).eigenenergies()
nzvals=vals[vals != 0]
# print(nzvals)

# m=np.arange(16).reshape([4,4])
# print(m)
# print(m[1][0])
# for i in range(13,12):
#     print(i)

# m=10
# coherencias={'0,10':[],'10,10':[]}
# for key in coherencias.keys():
#     if key.split(',')[0].startswith(str(m)) or key.split(',')[1].startswith(str(m)):
#         print('sape loquita')
# sqrtN=tensor(qeye(2),qeye(2),Qobj(np.diag([0,1,np.sqrt(2)])))
# print(n)   
# print('----sqrt n----')
# print(sqrtN)
# print('-----sqrt n tidyup----')
# print(sqrtN.tidyup())

