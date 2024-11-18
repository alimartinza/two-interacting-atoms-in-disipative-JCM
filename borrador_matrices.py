import numpy as np
import matplotlib.pyplot as plt


g=0.001
k=np.linspace(0,g,4)
delta=np.linspace(0,4*g,4)
kdeltazip=list((str(np.round(i[0]/g,2))+"g",str(np.round(i[1]/g,2))+"g") for i in zip(k,delta))
k_ax, delta_ax = np.meshgrid(k,
                delta,sparse=True)
frames=3
zs0=np.zeros((len(delta),len(k),frames))

zs0[:,:,0]=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
zs0[:,:,1]=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
zs0[:,:,2]=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]

zs0[:,:,0].flatten()
zs0[:,:,1].flatten()
zs0[:,:,2].flatten()
print(zs0)
zs0.reshape(-1,frames)
print(np.shape(zs0))
i=0
for data_t in zs0:
    print(i)
    print(data_t)
    i+=1
