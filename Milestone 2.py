"""
Ampliación de Matemáticas - Master Universitario en Sistemas Espaciales - ETSIAE
Milestone 2 : Prototypes to integrate orbits with functions.
1. Write a function called Euler to integrate one step. The function F(U, t)
of the Cauchy problem should be input argument.
2. Write a function called Crank_Nicolson to integrate one step.
3. Write a function called RK4 to integrate one step.
4. Write a function called Inverse_Euler to integrate one step.
5. Write a function to integrate a Cauchy problem. Temporal scheme, initial
condition and the function F(U, t) of the Cauchy problem should be input
arguments.
6. Write a function to express the force of the Kepler movement.
7. Integrate a Kepler with these latter schemes and explain the results.
8. Increase and decrease the time step and explained the results.
"""

from numpy import array,concatenate,zeros,abs,max
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def F(dr,r): #Definition of F(dr,-r/norm(r)**3). Transforms R4 to R4
    F1=dr
    F2=-r/norm(r)**3
    F_resultante=concatenate((F1,F2))
    
    return F_resultante

"""
----------------------------
1. FUNCTION EULER SCHEME
----------------------------
Euler scheme has an spectral radius p>1 so is unstable

Explicit method U_n+1 = U_n + delta_T * F(dr_n,r_n)
"""
def Euler(F,U0,n,delta_T):
    U_Euler=zeros((5,n)) #Definition of the size of U state vector where each row is [r(1x2),dr(1x2),t]'
    U_Euler[0:4,0]= U0 #Include initial conditions in U

    for i in range (n):
        U_Euler[4,i]=i*delta_T #Include time in U[5,:]

    for i in range (1,n):
        U_Euler[0:4,i]=U_Euler[0:4,i-1]+delta_T*F(U_Euler[2:4,i-1],U_Euler[0:2,i-1]) #Euler scheme--> U_n+1 = U_n + delta_T * F(dr_n,r_n)
    return U_Euler

"""
----------------------------
2. FUNCTION 1st ORDER CRANCK-NICOLSON SCHEME
----------------------------
Implicit method U_n+1 = U_n + delta_T/2 * (F(dr_n,r_n) + F(dr_n+1,r_n+1))

To solve the implicit method, first start with an Euler scheme U_n+1_Euler = U_n + delta_T * F(dr_n,r_n) and
introduce it in the Cranck_Nicolson scheme U_n+1_Cranck = U_n + delta_T/2 * (F(dr_n,r_n) + F(dr_n+1_Euler,r_n+1_Euler)) then
calculate the Error = U_n+1_Cranck - U_n+1_Euler and if the error is > tolerance calculate iteratively U_n+1_Cranck = U_n + delta_T/2 * (F(dr_n,r_n) + F(dr_n+1_Cranck_previous,r_n+1_Cranck_previous))
until the solution converges so the Error < tolerance
"""
def Cranck_Nicolson(F,U0,n,delta_T):
    U_Cranck=zeros((5,n)) #Definition of the size of U state vector where each row is [r(1x2),dr(1x2),t]'
    U_Cranck[0:4,0]= U0 #Include initial conditions in U
    y_Euler=zeros(4) #Definition of the vector in which the Euler solution will be stored for each time step to iterate
    y_Cranck=zeros((4,2)) #Definition of the vector in which the Cranck_Nicolson solution will be stored for each time step to iterate
    Error_Cranck=zeros(4) #Definition of the error
    
    for i in range (n):
        U_Cranck[4,i]=i*delta_T #Include time in U in U[5,:]

    for i in range (1,n):
        y_Euler=U_Cranck[0:4,i-1]+delta_T*F(U_Cranck[2:4,i-1],U_Cranck[0:2,i-1]) #Euler scheme--> U_n+1 = U_n + delta_T * F(dr_n,r_n)
        y_Cranck[0:4,0]=U_Cranck[0:4,i-1]+delta_T/2*(F(U_Cranck[2:4,i-1],U_Cranck[0:2,i-1])+F(y_Euler[2:4],y_Euler[0:2])) #Cranck-Nicolson scheme--> U_n+1 = U_n + delta_T/2 * (F(dr_n,r_n) + F(dr_n+1,r_n+1)) with Euler first
        Error_Cranck=y_Cranck[0:4,0]-y_Euler # Calculates the first error Cranck - Euler
        y_Cranck[0:4,1]=y_Cranck[0:4,0]
        while max(abs(Error_Cranck))>1e-6:
            y_Cranck[0:4,1]=U_Cranck[0:4,i-1]+delta_T/2*(F(U_Cranck[2:4,i-1],U_Cranck[0:2,i-1])+F(y_Cranck[2:4,0],y_Cranck[0:2,0])) #Iterate in the same way but using Cranck-Nicolson solution instead of the Euler one until the solution converges
            Error_Cranck=y_Cranck[0:4,1]-y_Cranck[0:4,0]
            y_Cranck[0:4,0]=y_Cranck[0:4,1]
        
        U_Cranck[0:4,i]=y_Cranck[0:4,1]
    return U_Cranck
    
"""
----------------------------
3. FUNCTION 4th ORDER EXPLICIT RUNGE-KUTTA SCHEME
----------------------------
Explicit method U_n+1 = U_n + delta_T/6 * (k1 + 2*k2 + 2*k3 + k4)
where   k1 = F(t_n, dr_n, r_n)
        k2 = F(t_n + delta_T/2, dr_n + delta_T/2*k1, r_n + delta_T/2*k1) 
        k3 = F(t_n + delta_T/2, dr_n + delta_T/2*k2, r_n + delta_T/2*k2)
        k4 = F(t_n + delta_T, dr_n + delta_T*k3, r_n + delta_T*k3)

"""
def RK4(F,U0,n,delta_T):
    U_Runge=zeros((5,n)) #Definition of the size of U state vector where each row is [r(1x2),dr(1x2),t]'
    U_Runge[0:4,0]= U0 #Include initial conditions in U
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
    
    return U_Runge

"""
----------------------------
4. FUNCTION INVERSE EULER SCHEME
----------------------------
Implicit method U_n+1 = U_n + delta_T* (F(dr_n+1,r_n+1))

To solve the implicit method, first start with an Euler scheme U_n+1_Euler = U_n + delta_T * F(dr_n,r_n) and
introduce it in the Inverse Euler scheme U_n+1_InverseEuler = U_n + delta_T * (F(dr_n+1_Euler,r_n+1_Euler)) then
calculate the Error = U_n+1_InverseEuler - U_n+1_Euler and if the error is > tolerance calculate iteratively U_n+1_InverseEuler = U_n + delta_T* ( F(dr_n+1_InverseEuler_previous,r_n+1_InverseEuler_previous))
until the solution converges so the Error < tolerance
"""

def Inverse_Euler(F,U0,n,delta_T):
    U_InverseEuler=zeros((5,n)) #Definition of the size of U state vector where each row is [r(1x2),dr(1x2),t]'
    U_InverseEuler[0:4,0]= U0 #Include initial conditions in U
    y_Euler=zeros(4) #Definition of the vector in which the Euler solution will be stored for each time step to iterate
    y_InverseEuler=zeros((4,2)) #Definition of the vector in which the InverseEuler solution will be stored for each time step to iterate
    Error_InverseEuler=zeros(4) #Definition of the error
    
    for i in range (n):
        U_InverseEuler[4,i]=i*delta_T #Include time in U in U[5,:]

    for i in range (1,n):
        y_Euler=U_InverseEuler[0:4,i-1]+delta_T*F(U_InverseEuler[2:4,i-1],U_InverseEuler[0:2,i-1]) #Euler scheme--> U_n+1 = U_n + delta_T * F(dr_n,r_n)
        y_InverseEuler[0:4,0]=U_InverseEuler[0:4,i-1]+delta_T*(F(y_Euler[2:4],y_Euler[0:2])) #InverseEuler scheme--> U_n+1 = U_n + delta_T * ( F(dr_n+1,r_n+1)) with Euler first
        Error_InverseEuler=y_InverseEuler[0:4,0]-y_Euler # Calculates the first error InverseEuler - Euler
        y_InverseEuler[0:4,1]=y_InverseEuler[0:4,0]
        while max(abs(Error_InverseEuler))>1e-6:
            y_InverseEuler[0:4,1]=U_InverseEuler[0:4,i-1]+delta_T*(F(y_InverseEuler[2:4,0],y_InverseEuler[0:2,0])) #Iterate in the same way but using Cranck-Nicolson solution instead of the Euler one until the solution converges
            Error_InverseEuler=y_InverseEuler[0:4,1]-y_InverseEuler[0:4,0]
            y_InverseEuler[0:4,0]=y_InverseEuler[0:4,1]
        
        U_InverseEuler[0:4,i]=y_InverseEuler[0:4,1]
    return U_InverseEuler

"""
----------------------------
5. FUNCTION CAUCHY PROBLEM
----------------------------
Function to integrate the Cauchy problem selecting the temporal scheme

"""
def Cauchy(Temporal_Scheme,F,U0,n,delta_T):
    U=Temporal_Scheme(F,U0,n,delta_T)
    return U
"""
----------------------------
6. FUNCTION KEPLER FORCE
----------------------------
Function to write the Kepler force instead of F. In this case the same as F

"""
def Kepler_Force(dr,r): #Definition of F(dr,-r/norm(r)**3). Transforms R4 to R4
    F1=dr
    F2=-r/norm(r)**3
    F_resultante=concatenate((F1,F2))
    
    return F_resultante


r0 = array([1, 0]) #Definition of initial position
dr0 = array([0, 1]) #Definition of initial velocity
delta_T=0.1 #Definition of Delta T = t_total/n
n=1000 #Definition of n number of steps
U0=concatenate((r0,dr0)) #initial conditions U0 

U_Euler=Cauchy(Euler,Kepler_Force,U0,n,delta_T)
U_Cranck=Cauchy(Cranck_Nicolson,Kepler_Force,U0,n,delta_T) #Needs bigger delta_T to converge in the Kepler problem
U_Runge=Cauchy(RK4,Kepler_Force,U0,n,delta_T)
Factor=100 #For lowering delta_T to help converge the Inverse Euler scheme in the Kepler problem
U_InverseEuler=Cauchy(Inverse_Euler,Kepler_Force,U0,n*Factor,delta_T/Factor) #Needs lower delta_T to converge in the Kepler problem


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

x_InverseEuler = U_InverseEuler[0, ::Factor] # Gets a component of the U_InverseEuler every Factor position
y_InverseEuler = U_InverseEuler[1, ::Factor]


plt.figure(figsize=(8, 6))
plt.plot(x_Euler, y_Euler, label='Euler')
plt.plot(x_Cranck, y_Cranck, label='Cranck-Nicolson')
plt.plot(x_Runge, y_Runge, label='Runge-Kutta 4th Order')
plt.plot(x_InverseEuler, y_InverseEuler, label='Inverse Euler')
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
ax.plot(x_InverseEuler, y_InverseEuler, t, label='Inverse Euler')
ax.scatter(x_Euler[0], y_Euler[0], t[0], color='red', s=100, label='Initial position')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('t')
ax.set_title('Kepler orbit trajectory with time')
ax.legend()
plt.show()