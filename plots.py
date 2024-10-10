import jcm_lib 

w0=1
J=0
g=0.001*w0
k=0.1*g
p=0.005*g
# t_final=25000
# steps=2000
# t=np.linspace(0,t_final,steps)
save_plot=False
plot_show=True
delta=[0,0.1*g,0.2*g,0.3*g,0.4*g,0.5*g,0.6*g,0.7*g,0.8*g,0.9*g,g,1.1*g,1.2*g,1.3*g,1.4*g,1.5*g,1.6*g,1.7*g,1.8*g,1.9*g,2*g]
chi=1.7*g
jcm_lib.anim2x2_delta('ee0+gg2',"10_3_9 unitario lineal",['pr(gg2)','pr(eg1+ge1)','pr(ee0)','FG'],delta,chi,0.1*g,saveword="fer")