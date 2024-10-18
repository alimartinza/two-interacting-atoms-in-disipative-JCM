import numpy as np
import matplotlib.pyplot as plt

g=0.001
k=np.linspace(0,g,7)
delta=np.linspace(0,4*g,7)
k_ax, delta_ax = np.meshgrid(k,
                delta,sparse=True)
zs=[[0,0,0,0,0,0,0],[1,1,1,1,1,1,1],2*np.ones(7),3*np.ones(7),4*np.ones(7),5*np.ones(7),6*np.ones(7)]

h=plt.contourf(k,delta,zs)
plt.xlabel('k')
plt.ylabel('delta')
plt.axis('scaled')
plt.colorbar()
plt.show()