import sys, os, numpy as np
# sys.path.insert(0,'/mnt/project'); os.chdir('/mnt/project')
import warnings; warnings.filterwarnings('ignore')
from qutip import basis, tensor, qeye, destroy, num, sigmam, sigmap, sigmaz, Qobj, mesolve
from jcm_lib import fases

N_c=3
e=basis(2,0); gr=basis(2,1)
n2=tensor(qeye(2),qeye(2),Qobj(np.diag([i*i for i in range(N_c)])))
a=tensor(qeye(2),qeye(2),destroy(N_c))
sm1=tensor(sigmam(),qeye(2),qeye(N_c)); sp1=tensor(sigmap(),qeye(2),qeye(N_c)); sz1=tensor(sigmaz(),qeye(2),qeye(N_c))
sm2=tensor(qeye(2),sigmam(),qeye(N_c)); sp2=tensor(qeye(2),sigmap(),qeye(N_c)); sz2=tensor(qeye(2),sigmaz(),qeye(N_c))
ee0=tensor(e,e,basis(N_c,0)); eg1=tensor(e,gr,basis(N_c,1)); ge1=tensor(gr,e,basis(N_c,1)); gg2=tensor(gr,gr,basis(N_c,2))
def H_tcm(g,d,x,k,J):
    return x*n2 + d/2*(sz1+sz2) + g*((sm1+sm2)*a.dag()+(sp1+sp2)*a) + 2*k*(sm1*sp2+sp1*sm2) + J*sz1*sz2
def l_ops(gamma,p):
    return [np.sqrt(gamma)*a,np.sqrt(p)*sm1,np.sqrt(p)*sm2]
def wrap(x): return (x+np.pi)%(2*np.pi)-np.pi

g=0.01
phi1,phi2,phi3=ee0,(eg1+ge1).unit(),gg2
block=[phi1,phi2,phi3]
def block_H(D):
    nn=2; return np.array([[D,np.sqrt(2*(nn-1))*g,0],[np.sqrt(2*(nn-1))*g,0,np.sqrt(2*nn)*g],[0,np.sqrt(2*nn)*g,-D]])

e1=np.array([1,-1,0])/np.sqrt(2); e2=np.array([1,1,-2])/np.sqrt(6); o=np.array([1/3,1/3,1/3])
r2=np.sqrt(1/2-1/3)
def pops_theta(th): return o+r2*(np.cos(th)*e1+np.sin(th)*e2)
def psi0_theta(th,evecs):
    p=pops_theta(th)
    if np.any(p<-1e-9): return None
    p=np.clip(p,0,None); c=np.sqrt(p)
    pb=np.zeros(3,dtype=complex)
    for k in range(3): pb+=c[k]*evecs[:,k]
    return (pb[0]*phi1+pb[1]*phi2+pb[2]*phi3).unit()

def dphi(psi0,D,gamma,p_at,m=2,spc=1200):
    ev=np.linalg.eigvalsh(block_H(D))
    Omega=min(abs(ev[1]-ev[0]),abs(ev[2]-ev[0]),abs(ev[2]-ev[1]))
    if Omega<1e-9: return np.nan
    T=2*np.pi/Omega; t=np.linspace(0,m*T,m*spc)
    su=mesolve(H_tcm(g,D,0,0,0),psi0,t,c_ops=[])
    sd=mesolve(H_tcm(g,D,0,0,0),psi0,t,c_ops=l_ops(gamma,p_at))
    fu,_=fases(su,open_system=False); fd,_,_,_=fases(sd,open_system=True,N1=4)
    return wrap(fd[-1]-fu[-1])

if __name__=="__main__":
    gamma=0.1*g; p_at=0.005*g
    N_D, N_th = 50, 50
    D_arr=np.linspace(0,6*g,N_D); th_arr=np.linspace(0,2*np.pi,N_th)
    Z=np.full((N_th,N_D),np.nan); Neff=np.full((N_th,N_D),np.nan)
    for j,D in enumerate(D_arr):
        evals,evecs=np.linalg.eigh(block_H(D))
        for i,th in enumerate(th_arr):
            psi0=psi0_theta(th,evecs)
            if psi0 is None: continue
            c=evecs.conj().T@np.array([complex(b.overlap(psi0)) for b in block])
            pops=np.abs(c)**2; Neff[i,j]=1/np.sum(pops**2)
            try: Z[i,j]=dphi(psi0,D,gamma,p_at)
            except Exception as ex: print("fail",D/g,th,ex)
        print(f"D/g={D/g:5.2f} done  |dphi|=[{np.nanmin(np.abs(Z[:,j])):.3f},{np.nanmax(np.abs(Z[:,j])):.3f}]",flush=True)
        # np.savez("orbits/data/gp_Neff2_scan.npz",D_arr=D_arr,th_arr=th_arr,Z=Z,Neff=Neff,g=g)
    np.savez('orbits/data/gp_Neff2_scan.npz',D_arr=D_arr,th_arr=th_arr,Z=Z,Neff=Neff,g=g)
    print("saved")

