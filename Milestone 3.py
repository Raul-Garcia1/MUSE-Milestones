"""
Ampliación de Matemáticas - Master Universitario en Sistemas Espaciales - ETSIAE
Milestone 2 : Error estimation of numerical solutions.
 1. Write a function to evaluate errors of numerical integration by means of
 Richardson extrapolation. This function should be based on the Cauchy
 problem solution implemented in milestone 2.
 2. Numerical error or different temporal schemes: Euler, Inverse Euler, Crank
 Nicolson and fourth order Runge Kutta method.
 3. Write a function to evaluate the convergence rate of different temporal
 schemes.
 4. Convergence rate of the different methods with the time step.
"""

from numpy import array,concatenate,zeros,abs,max,log
from numpy.linalg import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import inspect

def Euler(F,U0,n,delta_T):
    U_Euler=zeros((5,n)) #Definition of the size of U state vector where each row is [r(1x2),dr(1x2),t]'
    U_Euler[0:4,0]= U0 #Include initial conditions in U

    for i in range (n):
        U_Euler[4,i]=i*delta_T #Include time in U[5,:]

    for i in range (1,n):
        U_Euler[0:4,i]=U_Euler[0:4,i-1]+delta_T*F(U_Euler[2:4,i-1],U_Euler[0:2,i-1]) #Euler scheme--> U_n+1 = U_n + delta_T * F(dr_n,r_n)
    return U_Euler

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

def Cauchy(Temporal_Scheme,F,U0,n,delta_T):
    U=Temporal_Scheme(F,U0,n,delta_T)
    return U

def Kepler_Force(dr,r): #Definition of F(dr,-r/norm(r)**3). Transforms R4 to R4
    F1=dr
    F2=-r/norm(r)**3
    F_resultante=concatenate((F1,F2))
    
    return F_resultante

"""
----------------------------
1. ERROR ESTIMATION USING RICHARDSON EXTRAPOLATION FUNCTION
----------------------------
Richardson extrapolation: phi = phi(h) + c1*h^p + c2*h^p... where phi is the exact solution, phi(h) is an aproximation using a step of size h, and c1,c2... are ctes unknown
The thing is to eliminate the terms with ctes using two different time steps: h and h/r
Finally you reach to: phi=(phi(h)-r^p*phi(h/r))/(1-r^p)

The ERROR with a step size h/r is defined as E(h/r) = phi - phi(h/r)
Introducing the Richarson extrapolation: E(h/r) = (phi(h)-r^p*phi(h/r))/(1-r^p) - phi(h/r)

The order of p for the different schemes:
- Euler/Inverse Euler: p=1
- Cranck-Nicolson: p=2
- RK4: p=4
"""
def Error_Richardson_Extrapolation(Temporal_Scheme,F,U0,n,delta_T):
    r=2 # r is typically defined as 2 but could be another
    Error=zeros((5,(n-1)//r+1))
    phi_h=Temporal_Scheme(F,U0,(n-1)//r+1,delta_T*r)
    phi_h_r=Temporal_Scheme(F,U0,n,delta_T)
    phi_h_r_same_dimension=phi_h_r[:,::r]
    
    scheme_name = Temporal_Scheme.__name__
    
    if scheme_name == 'Euler':
        p = 1
    elif scheme_name == 'Inverse_Euler':
        p = 1
    elif scheme_name == 'Cranck_Nicolson':
        p = 2
    elif scheme_name == 'RK4':
        p = 4
    else:
        raise ValueError(f"Esquema temporal '{scheme_name}' no reconocido.")

    for i in range (0,4):
        Error[i,:]= (phi_h[i,:]-r**p*phi_h_r_same_dimension[i,:])/(1-r**p) - phi_h_r_same_dimension[i,:]
    
    for i in range ((n-1)//r+1):
        Error[4,i]=i*delta_T*r #Include time in Error[5,:]

    return Error

"""
----------------------------
3. CONVERGENCE RATE FUNCTION
----------------------------
Used to verify the scheme problem is being well calculated. It calculates the order p of the scheme that has to be:
- Euler/Inverse Euler: p=1
- Cranck-Nicolson: p=2
- RK4: p=4

The error E can be written in terms of the step size delta_T: E = C * delta^p where C is a cte
In terms of log: log E = p * log delta_T + log C which is a straight line
So if we calculate the Error at the same time T but using two different delta_T: E1, delta_T1, E2, delta_T2
p will be slope p=(log E2 - log E1)/(delta_T2 - delta_T1)

"""    
def Convergence_Rate(Temporal_Scheme,Temporal_Scheme_for_exact_solution,F,U0,n,delta_T):

    U_exacto=Temporal_Scheme_for_exact_solution(F,U0,(n-1)*1000+1,delta_T/1000) #n-1 is used to mstch the final t, as the number of steps is n-1
    U1=Temporal_Scheme(F,U0,(n-1)*2+1,delta_T/2)
    U2=Temporal_Scheme(F,U0,(n-1)+1,delta_T)

    E1=U_exacto[0:4,-1]-U1[0:4,-1]
    E2=U_exacto[0:4,-1]-U2[0:4,-1]
    
    E1=norm(E1)
    E2=norm(E2)
    p=(log(E2) - log(E1))/(log(delta_T) - log(delta_T/2))
    return p

r0 = array([1, 0]) #Definition of initial position
dr0 = array([0, 1]) #Definition of initial velocity
delta_T=0.001 #Definition of Delta T = t_total/n
n=50000 #Definition of n number of steps
U0=concatenate((r0,dr0)) #initial conditions U0 

Error_Richardson_Extrapolation_Euler1=Error_Richardson_Extrapolation(Euler,Kepler_Force,U0,n,delta_T)
Error_Richardson_Extrapolation_Cranck1=Error_Richardson_Extrapolation(Cranck_Nicolson,Kepler_Force,U0,n,delta_T) 
Error_Richardson_Extrapolation_Runge1=Error_Richardson_Extrapolation(RK4,Kepler_Force,U0,n,delta_T)
Error_Richardson_Extrapolation_InverseEuler2=Error_Richardson_Extrapolation(Inverse_Euler,Kepler_Force,U0,n,delta_T) #Needs lower delta_T to converge in the Kepler problem

p_RK4=Convergence_Rate(RK4,RK4,Kepler_Force,U0,100,0.01)
print("p_RK4 = ",p_RK4)
p_Euler=Convergence_Rate(Euler,RK4,Kepler_Force,U0,100,0.01)
print("p_Euler = ",p_Euler)
p_InverseEuler=Convergence_Rate(Inverse_Euler,RK4,Kepler_Force,U0,100,0.01)
print("p_InverseEuler = ",p_InverseEuler)
p_Cranck=Convergence_Rate(Cranck_Nicolson,RK4,Kepler_Force,U0,100,0.01)
print("p_CranckNicolson = ",p_Cranck)

"""
----------------------------
2D GRAPHS
----------------------------
"""

# --- 2D GRAPH ---

x_Euler1 = Error_Richardson_Extrapolation_Euler1[0, :]
y_Euler1 = Error_Richardson_Extrapolation_Euler1[1, :]
dx_Euler1 = Error_Richardson_Extrapolation_Euler1[2, :]
dy_Euler1 = Error_Richardson_Extrapolation_Euler1[3, :]

x_Cranck1 = Error_Richardson_Extrapolation_Cranck1[0, :]
y_Cranck1 = Error_Richardson_Extrapolation_Cranck1[1, :]
dx_Cranck1 = Error_Richardson_Extrapolation_Cranck1[2, :]
dy_Cranck1 = Error_Richardson_Extrapolation_Cranck1[3, :]

x_Runge1 = Error_Richardson_Extrapolation_Runge1[0, :]
y_Runge1 = Error_Richardson_Extrapolation_Runge1[1, :]
dx_Runge1 = Error_Richardson_Extrapolation_Runge1[2, :]
dy_Runge1 = Error_Richardson_Extrapolation_Runge1[3, :]

x_InverseEuler2 = Error_Richardson_Extrapolation_InverseEuler2[0, :]
y_InverseEuler2 = Error_Richardson_Extrapolation_InverseEuler2[1, :]
dx_InverseEuler2 = Error_Richardson_Extrapolation_InverseEuler2[2, :]
dy_InverseEuler2 = Error_Richardson_Extrapolation_InverseEuler2[3, :]

t1=Error_Richardson_Extrapolation_Euler1[4, :]



plt.figure(figsize=(8, 6))
plt.plot(t1, x_Euler1, label='Euler')
plt.plot(t1, x_Cranck1, label='Cranck-Nicolson')
plt.plot(t1, x_InverseEuler2, label='Inverse Euler')
plt.plot(t1, x_Runge1, label='Runge-Kutta 4th Order')
plt.xlabel('t')
plt.ylabel('X')
plt.title(f'Error with Richardson Extrapolation in x (with $\\Delta t$ = {delta_T:.4f})')
plt.grid(True) 
plt.legend()  
plt.show() 

plt.figure(figsize=(8, 6))
plt.plot(t1, y_Euler1, label='Euler')
plt.plot(t1, y_Cranck1, label='Cranck-Nicolson')
plt.plot(t1, y_InverseEuler2, label='Inverse Euler')
plt.plot(t1, y_Runge1, label='Runge-Kutta 4th Order')
plt.xlabel('t')
plt.ylabel('Y')
plt.title(f'Error with Richardson Extrapolation in y (with $\\Delta t$ = {delta_T:.4f})')
plt.grid(True) 
plt.legend()  
plt.show() 

plt.figure(figsize=(8, 6))
plt.plot(t1, dx_Euler1, label='Euler')
plt.plot(t1, dx_Cranck1, label='Cranck-Nicolson')
plt.plot(t1, dx_InverseEuler2, label='Inverse Euler')
plt.plot(t1, dx_Runge1, label='Runge-Kutta 4th Order')
plt.xlabel('t')
plt.ylabel('dx/dt')
plt.title(f'Error with Richardson Extrapolation in dx/dt (with $\\Delta t$ = {delta_T:.4f})')
plt.grid(True) 
plt.legend()  
plt.show() 

plt.figure(figsize=(8, 6))
plt.plot(t1, dy_Euler1, label='Euler')
plt.plot(t1, dy_Cranck1, label='Cranck-Nicolson')
plt.plot(t1, dy_InverseEuler2, label='Inverse Euler')
plt.plot(t1, dy_Runge1, label='Runge-Kutta 4th Order')
plt.xlabel('t')
plt.ylabel('dy/dt')
plt.title(f'Error with Richardson Extrapolation in dy/dt (with $\\Delta t$ = {delta_T:.4f})')
plt.grid(True) 
plt.legend()  
plt.show() 

