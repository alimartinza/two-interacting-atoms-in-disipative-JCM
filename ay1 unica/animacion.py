import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from matplotlib.animation import FuncAnimation

#primero definimos los datos del problema
m1=1
m2=1
v1_0=10
l0=1
k=1
M=m1+m2
mu=m1*m2/M
# con esto podemos decir que desde el CM 
r1_0=l0*m2/M
E0=1/2*m1*v1_0**2*(1+m1/m2)+1/2*k*l0**2

#defino unos parametros a y b para aligerar notacion


#hasta ahora todo lo que hicimos fue definir numeritos, nada depende del tiempo.

def r(t:list):
    in_sqrt1 =E0**2/m1**2-v1_0**2*l0**2*k/mu
    if in_sqrt1 < 0:
        print('valor invalido en sqrt1')
    
    return np.sqrt(mu**2/(k*m1)*(E0/m1+np.sqrt(in_sqrt1)*np.cos(2*np.sqrt(k/mu)*t)))

def w(r:list):
    return -mu/m1*v1_0*l0/r**2


# Parameters and initial conditions
num_points = 5000
t_final=20
t = np.linspace(0,t_final,num_points)
r1=r(t)
r2=m1/m2*r1
theta=cumulative_trapezoid(w(r1),dx=t_final/num_points,initial=0)
theta2=theta+np.pi



plt.figure(figsize=(8, 5))
plt.plot(t, r1)
plt.xlabel('t')
plt.ylabel('r(t)')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(t, theta)
plt.xlabel('t')
plt.ylabel(r'$\theta(t)$')
plt.legend()
plt.grid(True)
plt.show()

a=mu/m1*E0
b=-mu**2/(2*m1)*v1_0**2*l0**2
c=-k*m1/mu

def r_tita(tita):
    return 1/np.sqrt(1/(2*b)*np.sqrt(a**2-4*b*c)*np.cos(tita)+mu*E0/(2*m1*b))

# Set up the figure and polar axes
tita=np.linspace(0,4*np.pi,10000)
r_tra=r(tita)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='polar')
ax.set_rmax(np.max(r_tra) + 1)  # Set radial limit slightly beyond max r
ax.plot(tita,r_tra,'b-')
ax.plot(tita+np.pi,m1/m2*r_tra,'r-')
ax.grid(True)
ax.set_title("Trayectoria", va='bottom')

plt.show()

# Set up the figure and polar axes
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='polar')
ax.set_rmax(np.max([np.max(r1),np.max(r2)]) + 1)  # Set radial limit slightly beyond max r
ax.grid(True)
ax.set_title("Trayectoria", va='bottom')

# Initialize plots
line1, = ax.plot([], [], 'b-', alpha=0.7)  # Line for all previous points
current_point1, = ax.plot([], [], 'bo', markersize=8)  # Red circle for current point

line2, = ax.plot([], [], 'r-', alpha=0.7)  # Line for all previous points
current_point2, = ax.plot([], [], 'ro', markersize=8)  # Red circle for current point

# Animation update function
def update(frame):
    # Update the line with all points up to the current frame
    line1.set_data(theta[:frame+1], r1[:frame+1])
    line2.set_data(theta2[:frame+1], r2[:frame+1])
    
    # Update the current point (last point in the sequence)
    if frame >= 0:
        current_point1.set_data([theta[frame]], [r1[frame]])
        current_point2.set_data([theta2[frame]], [r2[frame]])
    
    return line1,line2, current_point1, current_point2

# Create animation
ani = FuncAnimation(fig, update, frames=num_points, interval=0.1, blit=True)

plt.tight_layout()
plt.show()

# To save the animation (uncomment if needed)
# ani.save('polar_animation.gif', writer='pillow', fps=20)