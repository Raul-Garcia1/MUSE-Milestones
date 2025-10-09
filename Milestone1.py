"""
Ampliación de Matemáticas - Master Universitario en Sistemas Espaciales - ETSIAE
Milestone 1 : Prototypes to integrate orbits without functions
1. Write a script to integrate Kepler orbits with an Euler method.
2. Write a script to integrate Kepler orbits with a Crank-Nicolson method.
3. Write a script to integrate Kepler orbits with a Runge-Kutta fourth order.
4. Change time step and plot orbits. Discuss results
"""
from numpy import array,concatenate,zeros,abs,max
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def F(dr,r): #Definition of F(dr,-r/norm(r)**3). Transforms R4 to R4
    F1=dr
    F2=-r/norm(r)**3
    F=concatenate((F1,F2))
    return F


r0 = array([1, 0]) #Definition of initial position
dr0 = array([0, 1]) #Definition of initial velocity
delta_T=0.1 #Definition of Delta T = t_total/n
n=1000 #Definition of n number of steps


"""
----------------------------
1. RESOLUTION WITH EULER SCHEME
----------------------------
Euler scheme has an spectral radius p>1 so is unstable

Explicit method U_n+1 = U_n + delta_T * F(dr_n,r_n)
"""
U_Euler=zeros((5,n)) #Definition of the size of U state vector where each column is [r(1x2),dr(1x2),t]'
U_Euler[0:4,0]= concatenate((r0,dr0)) #Include initial conditions in U

for i in range (n):
    U_Euler[4,i]=i*delta_T #Include time in U[5,:]

for i in range (1,n):
    U_Euler[0:4,i]=U_Euler[0:4,i-1]+delta_T*F(U_Euler[2:4,i-1],U_Euler[0:2,i-1]) #Euler scheme--> U_n+1 = U_n + delta_T * F(dr_n,r_n)


"""
----------------------------
2. RESOLUTION WITH 1st ORDER CRANCK-NICOLSON SCHEME
----------------------------
Implicit method U_n+1 = U_n + delta_T/2 * (F(dr_n,r_n) + F(dr_n+1,r_n+1))

To solve the implicit method, first start with an Euler scheme U_n+1_Euler = U_n + delta_T * F(dr_n,r_n) and
introduce it in the Cranck_Nicolson scheme U_n+1_Cranck = U_n + delta_T/2 * (F(dr_n,r_n) + F(dr_n+1_Euler,r_n+1_Euler)) then
calculate the Error = U_n+1_Cranck - U_n+1_Euler and if the error is > tolerance calculate iteratively U_n+1_Cranck = U_n + delta_T/2 * (F(dr_n,r_n) + F(dr_n+1_Cranck_previous,r_n+1_Cranck_previous))
until the solution converges so the Error < tolerance
"""
U_Cranck=zeros((5,n)) #Definition of the size of U state vector where each column is [r(1x2),dr(1x2),t]'
U_Cranck[0:4,0]= concatenate((r0,dr0)) #Include initial conditions in U
y_Euler=zeros(4) #Definition of the vector in which the Euler solution will be stored for each time step to iterate
y_Cranck=zeros((4,2)) #Definition of the vector in which the Cranck_Nicolson solution will be stored for each time step to iterate
Error_Cranck=zeros(4) #Definition of the error

for i in range (n):
    U_Cranck[4,i]=i*delta_T #Include time in U in U[5,:]

for i in range (1,n):
    y_Euler=U_Cranck[0:4,i-1]+delta_T*F(U_Cranck[2:4,i-1],U_Cranck[0:2,i-1]) #Euler scheme--> U_n+1 = U_n + delta_T * F(dr_n,r_n)
    y_Cranck[0:4,0]=U_Cranck[0:4,i-1]+delta_T/2*(F(U_Cranck[2:4,i-1],U_Cranck[0:2,i-1])+F(y_Euler[2:4],y_Euler[0:2])) #Cranck-Nicolson scheme--> U_n+1 = U_n + delta_T/2 * (F(dr_n,r_n) + F(dr_n+1,r_n+1)) with Euler first
    Error_Cranck=y_Cranck[0:4,0]-y_Euler # Calculates the first error Cranck - Euler

    while max(abs(Error_Cranck))>1e-6:
        y_Cranck[0:4,1]=U_Cranck[0:4,i-1]+delta_T/2*(F(U_Cranck[2:4,i-1],U_Cranck[0:2,i-1])+F(y_Cranck[2:4,0],y_Cranck[0:2,0])) #Iterate in the same way but using Cranck-Nicolson solution instead of the Euler one until the solution converges
        Error_Cranck=y_Cranck[0:4,1]-y_Cranck[0:4,0]
        y_Cranck[0:4,0]=y_Cranck[0:4,1]
        
    U_Cranck[0:4,i]=y_Cranck[0:4,1]
    


"""
----------------------------
3. RESOLUTION WITH 4th ORDER EXPLICIT RUNGE-KUTTA SCHEME
----------------------------
Explicit method U_n+1 = U_n + delta_T/6 * (k1 + 2*k2 + 2*k3 + k4)
where   k1 = F(t_n, dr_n, r_n)
        k2 = F(t_n + delta_T/2, dr_n + delta_T/2*k1, r_n + delta_T/2*k1) 
        k3 = F(t_n + delta_T/2, dr_n + delta_T/2*k2, r_n + delta_T/2*k2)
        k4 = F(t_n + delta_T, dr_n + delta_T*k3, r_n + delta_T*k3)

"""
U_Runge=zeros((5,n)) #Definition of the size of U state vector where each column is [r(1x2),dr(1x2),t]'
U_Runge[0:4,0]= concatenate((r0,dr0)) #Include initial conditions in U
k1=zeros(4)
k2=zeros(4)
k3=zeros(4)
k4=zeros(4)

for i in range (n):
    U_Runge[4,i]=i*delta_T #Include time in U[5,:]

for i in range (1,n):
    k1=F(U_Runge[2:4,i-1],U_Runge[0:2,i-1]) #k1 = F(t_n, dr_n, r_n)
    k2=F(U_Runge[2:4,i-1] + delta_T/2*k1[2:4],U_Runge[0:2,i-1] + delta_T/2*k1[0:2]) #k2 = F(t_n + delta_T/2, dr_n + delta_T/2*k1, r_n + delta_T/2*k1) 
    k3=F(U_Runge[2:4,i-1] + delta_T/2*k2[2:4],U_Runge[0:2,i-1] + delta_T/2*k2[0:2]) #k3 = F(t_n + delta_T/2, dr_n + delta_T/2*k2, r_n + delta_T/2*k2)
    k4=F(U_Runge[2:4,i-1] + delta_T*k3[2:4],U_Runge[0:2,i-1] + delta_T*k3[0:2]) #k4 = F(t_n + delta_T, dr_n + delta_T*k3, r_n + delta_T*k3)
    U_Runge[0:4,i]= U_Runge[0:4,i-1] + delta_T/6 * (k1 + 2*k2 + 2*k3 + k4) #Explicit Runge-Kutta 4th scheme--> U_n+1 = U_n + delta_T/6 * (k1 + 2*k2 + 2*k3 + k4)


"""
----------------------------
2D AND 3D GRAPHS
----------------------------
"""

# --- 2D GRAPH ---

x_Euler = U_Euler[0, :]
y_Euler = U_Euler[1, :]

x_Cranck = U_Cranck[0, :]
y_Cranck = U_Cranck[1, :]

x_Runge = U_Runge[0, :]
y_Runge = U_Runge[1, :]


plt.figure(figsize=(8, 6))
plt.plot(x_Euler, y_Euler, label='Euler')
plt.plot(x_Cranck, y_Cranck, label='Cranck-Nicolson')
plt.plot(x_Runge, y_Runge, label='Runge-Kutta 4th Order')
plt.scatter(x_Euler[0], y_Euler[0], color='red', s=100, zorder=5, label='Initial position')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kepler orbit trajectory')
plt.grid(True) 
plt.legend()  
plt.axis('equal') 
plt.show() 


# --- 3D GRAPH ---

t = U_Euler[4, :] 

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_Euler, y_Euler, t, label='Euler')
ax.plot(x_Cranck, y_Cranck, t, label='Cranck-Nicolson')
ax.plot(x_Runge, y_Runge, t, label='Runge-Kutta 4th Order')
ax.scatter(x_Euler[0], y_Euler[0], t[0], color='red', s=100, label='Initial position')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('t')
ax.set_title('Kepler orbit trajectory with time')
ax.legend()
plt.show()