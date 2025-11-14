"""
Ampliación de Matemáticas - Master Universitario en Sistemas Espaciales - ETSIAE
Milestone 4 : Linear problems. Regions of absolute stability.
1. Integrate the linear oscillator ¨x+x = 0 with some initial conditions. Use
 Euler, Inverse Euler, Leap Frog, Crank Nicolson and fourth order Runge
 Kutta method.
 2. Regions of absolute stability of the above methods.
 3. Explain the numerical results based on regions of absolute stability.
"""

from numpy import array,concatenate,zeros,abs,max,log,real,imag,isclose,eye,block,all,meshgrid,linspace,finfo,copy
from numpy.linalg import norm,eigvals
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import inspect

def Euler(F,U0,n,delta_T):
    dim=len(U0)
    U_Euler=zeros((dim+1,n)) #Definition of the size of U state vector where each row is [r(1x2),dr(1x2),t]'
    U_Euler[0:dim,0]= U0 #Include initial conditions in U

    for i in range (n):
        U_Euler[dim,i]=i*delta_T #Include time in U[5,:]

    for i in range (1,n):
        U_Euler[0:dim,i]=U_Euler[0:dim,i-1]+delta_T*F(U_Euler[dim//2:dim,i-1],U_Euler[0:dim//2,i-1]) #Euler scheme--> U_n+1 = U_n + delta_T * F(dr_n,r_n)
    return U_Euler

def Cranck_Nicolson(F,U0,n,delta_T):
    dim=len(U0)
    U_Cranck=zeros((dim+1,n)) #Definition of the size of U state vector where each row is [r(1x2),dr(1x2),t]'
    U_Cranck[0:dim,0]= U0 #Include initial conditions in U
    y_Euler=zeros(dim) #Definition of the vector in which the Euler solution will be stored for each time step to iterate
    y_Cranck=zeros((dim,2)) #Definition of the vector in which the Cranck_Nicolson solution will be stored for each time step to iterate
    Error_Cranck=zeros(dim) #Definition of the error
    
    for i in range (n):
        U_Cranck[dim,i]=i*delta_T #Include time in U in U[5,:]

    for i in range (1,n):
        y_Euler=U_Cranck[0:dim,i-1]+delta_T*F(U_Cranck[dim//2:dim,i-1],U_Cranck[0:dim//2,i-1]) #Euler scheme--> U_n+1 = U_n + delta_T * F(dr_n,r_n)
        y_Cranck[0:dim,0]=U_Cranck[0:dim,i-1]+delta_T/2*(F(U_Cranck[dim//2:dim,i-1],U_Cranck[0:dim//2,i-1])+F(y_Euler[dim//2:dim],y_Euler[0:dim//2])) #Cranck-Nicolson scheme--> U_n+1 = U_n + delta_T/2 * (F(dr_n,r_n) + F(dr_n+1,r_n+1)) with Euler first
        Error_Cranck=y_Cranck[0:dim,0]-y_Euler # Calculates the first error Cranck - Euler
        y_Cranck[0:dim,1]=y_Cranck[0:dim,0]
        while max(abs(Error_Cranck))>1e-6:
            y_Cranck[0:dim,1]=U_Cranck[0:dim,i-1]+delta_T/2*(F(U_Cranck[dim//2:dim,i-1],U_Cranck[0:dim//2,i-1])+F(y_Cranck[dim//2:dim,0],y_Cranck[0:dim//2,0])) #Iterate in the same way but using Cranck-Nicolson solution instead of the Euler one until the solution converges
            Error_Cranck=y_Cranck[0:dim,1]-y_Cranck[0:dim,0]
            y_Cranck[0:dim,0]=y_Cranck[0:dim,1]
        
        U_Cranck[0:dim,i]=y_Cranck[0:dim,1]
    return U_Cranck
    
def RK4(F,U0,n,delta_T):
    dim=len(U0)
    U_Runge=zeros((dim+1,n)) #Definition of the size of U state vector where each row is [r(1x2),dr(1x2),t]'
    U_Runge[0:dim,0]= U0 #Include initial conditions in U
    k1=zeros(dim)
    k2=zeros(dim)
    k3=zeros(dim)
    k4=zeros(dim)

    for i in range (n):
        U_Runge[dim,i]=i*delta_T #Include time in U[5,:]

    for i in range (1,n):
        k1=F(U_Runge[dim//2:dim,i-1],U_Runge[0:dim//2,i-1]) #k1 = F(t_n, dr_n, r_n)
        k2=F(U_Runge[dim//2:dim,i-1] + delta_T/2*k1[dim//2:dim],U_Runge[0:dim//2,i-1] + delta_T/2*k1[0:dim//2]) #k2 = F(t_n + delta_T/2, dr_n + delta_T/2*k1, r_n + delta_T/2*k1) 
        k3=F(U_Runge[dim//2:dim,i-1] + delta_T/2*k2[dim//2:dim],U_Runge[0:dim//2,i-1] + delta_T/2*k2[0:dim//2]) #k3 = F(t_n + delta_T/2, dr_n + delta_T/2*k2, r_n + delta_T/2*k2)
        k4=F(U_Runge[dim//2:dim,i-1] + delta_T*k3[dim//2:dim],U_Runge[0:dim//2,i-1] + delta_T*k3[0:dim//2]) #k4 = F(t_n + delta_T, dr_n + delta_T*k3, r_n + delta_T*k3)
        U_Runge[0:dim,i]= U_Runge[0:dim,i-1] + delta_T/6 * (k1 + 2*k2 + 2*k3 + k4) #Explicit Runge-Kutta 4th scheme--> U_n+1 = U_n + delta_T/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    return U_Runge

def Inverse_Euler(F,U0,n,delta_T):
    dim=len(U0)
    U_InverseEuler=zeros((dim+1,n)) #Definition of the size of U state vector where each row is [r(1x2),dr(1x2),t]'
    U_InverseEuler[0:dim,0]= U0 #Include initial conditions in U
    y_Euler=zeros(dim) #Definition of the vector in which the Euler solution will be stored for each time step to iterate
    y_InverseEuler=zeros((dim,2)) #Definition of the vector in which the InverseEuler solution will be stored for each time step to iterate
    Error_InverseEuler=zeros(dim) #Definition of the error
    
    for i in range (n):
        U_InverseEuler[dim,i]=i*delta_T #Include time in U in U[5,:]

    for i in range (1,n):
        y_Euler=U_InverseEuler[0:dim,i-1]+delta_T*F(U_InverseEuler[dim//2:dim,i-1],U_InverseEuler[0:dim//2,i-1]) #Euler scheme--> U_n+1 = U_n + delta_T * F(dr_n,r_n)
        y_InverseEuler[0:dim,0]=U_InverseEuler[0:dim,i-1]+delta_T*(F(y_Euler[dim//2:dim],y_Euler[0:dim//2])) #InverseEuler scheme--> U_n+1 = U_n + delta_T * ( F(dr_n+1,r_n+1)) with Euler first
        Error_InverseEuler=y_InverseEuler[0:dim,0]-y_Euler # Calculates the first error InverseEuler - Euler
        y_InverseEuler[0:dim,1]=y_InverseEuler[0:dim,0]
        while max(abs(Error_InverseEuler))>1e-6:
            y_InverseEuler[0:dim,1]=U_InverseEuler[0:dim,i-1]+delta_T*(F(y_InverseEuler[dim//2:dim,0],y_InverseEuler[0:dim//2,0])) #Iterate in the same way but using Cranck-Nicolson solution instead of the Euler one until the solution converges
            Error_InverseEuler=y_InverseEuler[0:dim,1]-y_InverseEuler[0:dim,0]
            y_InverseEuler[0:dim,0]=y_InverseEuler[0:dim,1]
        
        U_InverseEuler[0:dim,i]=y_InverseEuler[0:dim,1]
    return U_InverseEuler

def Cauchy(Temporal_Scheme,F,U0,n,delta_T):
    U=Temporal_Scheme(F,U0,n,delta_T)
    return U

def Kepler_Force(dr,r): #Definition of F(dr,-r/norm(r)**3). Transforms R4 to R4
    F1=dr
    F2=-r/norm(r)**3
    F_resultante=concatenate((F1,F2))
    
    return F_resultante

def Error_Richardson_Extrapolation(Temporal_Scheme,F,U0,n,delta_T):
    dim=len(U0)
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

    for i in range (0,dim):
        Error[i,:]= (phi_h[i,:]-r**p*phi_h_r_same_dimension[i,:])/(1-r**p) - phi_h_r_same_dimension[i,:]
    
    for i in range ((n-1)//r+1):
        Error[dim,i]=i*delta_T*r #Include time in Error[5,:]

    return Error
 
def Convergence_Rate(Temporal_Scheme,Temporal_Scheme_for_exact_solution,F,U0,n,delta_T):
    dim=len(U0)
    U_exacto=Temporal_Scheme_for_exact_solution(F,U0,(n-1)*1000+1,delta_T/1000) #n-1 is used to mstch the final t, as the number of steps is n-1
    U1=Temporal_Scheme(F,U0,(n-1)*2+1,delta_T/2)
    U2=Temporal_Scheme(F,U0,(n-1)+1,delta_T)

    E1=U_exacto[0:dim,-1]-U1[0:dim,-1]
    E2=U_exacto[0:dim,-1]-U2[0:dim,-1]
    
    E1=norm(E1)
    E2=norm(E2)
    p=(log(E2) - log(E1))/(log(delta_T) - log(delta_T/2))
    return p

"""
----------------------------
0. LEAP FROG FUNCTION
----------------------------
Leap Frog scheme: U_n+1 = U_n-1 + 2* delta_T * F(dr_n,r_n)

Multi-step scheme that needs to start with another scheme. Euler scheme is often used for the first step

"""

def Leap_Frog(F,U0,n,delta_T):
    dim=len(U0)
    U_Leap=zeros((dim+1,n)) #Definition of the size of U state vector where each row is [r(1x2),dr(1x2),t]'
    U_Leap[0:dim,0]= U0 #Include initial conditions in U

    for i in range (n):
        U_Leap[dim,i]=i*delta_T #Include time in U[5,:]

    U_Leap[0:dim,1]=U_Leap[0:dim,0]+delta_T*F(U_Leap[dim//2:dim,0],U_Leap[0:dim//2,0]) # Initiates with an Euler

    for i in range (2,n):
        U_Leap[0:dim,i]=U_Leap[0:dim,i-2]+2*delta_T*F(U_Leap[dim//2:dim,i-1],U_Leap[0:dim//2,i-1]) #Leap Frog scheme--> U_n+1 = U_n-1 + 2* delta_T * F(dr_n,r_n)
    return U_Leap
"""
----------------------------
1. LINEAR OSCILLATOR FUNCTION
----------------------------
Linear oscillator ¨x+x = 0 equation

"""

def F(dr,r): #Definition of F(dr,d2r). Transforms R4 to R4
    F1=dr
    F2=-r
    F_resultante=concatenate((F1,F2))
    
    return F_resultante

"""
----------------------------
2. STABILITY FUNCTION FUNCTION
----------------------------
Stability regions for the different schemes (z=lambda*delta_T):
- Euler: |1+z| ≤ 1
- Inverse Euler: |1/(1-z)| ≤ 1
- Cranck Nicolson: Re(z) ≤ 0
- RK4: |1+z+z^2/2+z^3/6+z^4/24| ≤ 1
- Leap Frog: -|(1) Re(z) = 0
             -|(2) Im(z) ≤ 2

"""

def Jacobian(F,dr,r):
    dim=len(r)
    
    Jr=zeros((dim,dim)) #Definition of the size of the Jacobian dim x dim
    Jdr=zeros((dim,dim)) #Definition of the size of the Jacobian dim x dim
    epsilon=finfo(float).eps**0.5
    for i in range(dim):
        r_per_pos=copy(r)
        r_per_pos[i]=r_per_pos[i]+epsilon
        F_full_per_r_pos = F(dr, r_per_pos) 
        F_accel_per_r_pos = F_full_per_r_pos[dim : 2*dim]

        r_per_neg=copy(r)
        r_per_neg[i]=r_per_neg[i]-epsilon
        F_full_per_r_neg = F(dr, r_per_neg) 
        F_accel_per_r_neg = F_full_per_r_neg[dim : 2*dim]

        dr_per_pos=copy(dr)
        dr_per_pos[i]=dr_per_pos[i]+epsilon
        F_full_per_dr_pos = F(dr_per_pos, r) 
        F_accel_per_dr_pos = F_full_per_dr_pos[dim : 2*dim]

        dr_per_neg=copy(dr)
        dr_per_neg[i]=dr_per_neg[i]-epsilon
        F_full_per_dr_neg = F(dr_per_neg, r) 
        F_accel_per_dr_neg = F_full_per_dr_neg[dim : 2*dim]

        Jr[:, i] = (F_accel_per_r_pos - F_accel_per_r_neg) / (2*epsilon)
        Jdr[:, i] = (F_accel_per_dr_pos - F_accel_per_dr_neg) / (2*epsilon)
    return Jr,Jdr

def Linear_Matrix_A(Jacobian,F,dr,r):
    dim=len(r)
    A=zeros((2*dim,2*dim)) 
    Jr, Jdr = Jacobian(F, dr, r) 
    
    Zero = zeros((dim, dim))
    Identity = eye(dim)
    
    A = block([
        [Zero, Identity], 
        [Jr, Jdr]         
    ])
    return A

def Stability_Numeric(Linear_Matrix_A,Jacobian,Temporal_Scheme,F,Ueq,delta_T):
    dim=len(Ueq)//2
    r=Ueq[0:dim]
    dr=Ueq[dim:2*dim]
    A=Linear_Matrix_A(Jacobian,F,dr,r)
    eigenvalues = eigvals(A)
    z=eigenvalues*delta_T

    tolerance=1e-12
    scheme_name = Temporal_Scheme.__name__
    
    if scheme_name == 'Euler':
        Rz = abs(1+z)
        if all(Rz <= 1+tolerance):
            print(f"Esquema temporal '{scheme_name}' es ESTABLE para este problema con este delta_T. Autovalores=",A)
        else:    
            print(f"Esquema temporal '{scheme_name}' es INESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)

    elif scheme_name == 'Inverse_Euler':
        Rz = abs(1/(1-z))
        if all(Rz <= 1+tolerance):
            print(f"Esquema temporal '{scheme_name}' es ESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)
        else:    
            print(f"Esquema temporal '{scheme_name}' es INESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)

    elif scheme_name == 'Cranck_Nicolson':
        Rz = real(z)
        if all(Rz <= 0+tolerance):
            print(f"Esquema temporal '{scheme_name}' es ESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)
        else:    
            print(f"Esquema temporal '{scheme_name}' es INESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)

    elif scheme_name == 'RK4':
        Rz = abs(1+z+z**2/2+z**3/6+z**4/24)
        if all(Rz <= 1+tolerance):
            print(f"Esquema temporal '{scheme_name}' es ESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)
        else:    
            print(f"Esquema temporal '{scheme_name}' es INESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)

    elif scheme_name == 'Leap_Frog':
        Rz1 = real(z)
        Rz2 = abs(imag(z))
        if all(isclose(Rz1, 0, atol=tolerance)) and all(Rz2<=1+tolerance):
            print(f"Esquema temporal '{scheme_name}' es ESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)
        else:    
            print(f"Esquema temporal '{scheme_name}' es INESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)

    return 


def Stability(A,Temporal_Scheme,delta_T):
    eigenvalues = eigvals(A)
    z=eigenvalues*delta_T

    tolerance=1e-12
    scheme_name = Temporal_Scheme.__name__
    
    if scheme_name == 'Euler':
        Rz = abs(1+z)
        if all(Rz <= 1+tolerance):
            print(f"Esquema temporal '{scheme_name}' es ESTABLE para este problema con este delta_T. Autovalores=",eigenvalues)
        else:    
            print(f"Esquema temporal '{scheme_name}' es INESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)

    elif scheme_name == 'Inverse_Euler':
        Rz = abs(1/(1-z))
        if all(Rz <= 1+tolerance):
            print(f"Esquema temporal '{scheme_name}' es ESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)
        else:    
            print(f"Esquema temporal '{scheme_name}' es INESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)

    elif scheme_name == 'Cranck_Nicolson':
        Rz = real(z)
        if all(Rz <= 0+tolerance):
            print(f"Esquema temporal '{scheme_name}' es ESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)
        else:    
            print(f"Esquema temporal '{scheme_name}' es INESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)

    elif scheme_name == 'RK4':
        Rz = abs(1+z+z**2/2+z**3/6+z**4/24)
        if all(Rz <= 1+tolerance):
            print(f"Esquema temporal '{scheme_name}' es ESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)
        else:    
            print(f"Esquema temporal '{scheme_name}' es INESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)

    elif scheme_name == 'Leap_Frog':
        Rz1 = real(z)
        Rz2 = abs(imag(z))
        if all(isclose(Rz1, 0, atol=tolerance)) and all(Rz2<=1+tolerance):
            print(f"Esquema temporal '{scheme_name}' es ESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)
        else:    
            print(f"Esquema temporal '{scheme_name}' es INESTABLE para este problema con este delta_T.Autovalores=",eigenvalues)

    return z 

r0 = array([1,0]) #Definition of initial position
dr0 = array([0,1]) #Definition of initial velocity
delta_T=0.1 #Definition of Delta T = t_total/n
n=1000 #Definition of n number of steps
U0=concatenate((r0,dr0)) #initial conditions U0 

A= array([
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
    [-1., 0., 0., 0.],
    [0., -1., 0., 0.]]) #Specific of the linear oscillator

z=Stability(A,Euler,delta_T)
z=Stability(A,Inverse_Euler,delta_T)
z=Stability(A,Cranck_Nicolson,delta_T)
z=Stability(A,RK4,delta_T)
z=Stability(A,Leap_Frog,delta_T)

U_Euler=Cauchy(Euler,F,U0,n,delta_T)
U_Cranck=Cauchy(Cranck_Nicolson,F,U0,n,delta_T) 
U_Runge=Cauchy(RK4,F,U0,n,delta_T)
U_InverseEuler=Cauchy(Inverse_Euler,F,U0,n,delta_T) 
U_LeapFrog=Cauchy(Leap_Frog,F,U0,n,delta_T)

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

x_InverseEuler = U_InverseEuler[0, :]
y_InverseEuler = U_InverseEuler[1, :]

x_LeapFrog = U_LeapFrog[0, :]
y_LeapFrog = U_LeapFrog[1, :]

plt.figure(figsize=(8, 6))
plt.plot(x_Euler, y_Euler, label='Euler')
plt.plot(x_Cranck, y_Cranck, label='Cranck-Nicolson')
plt.plot(x_Runge, y_Runge, label='Runge-Kutta 4th Order')
plt.plot(x_InverseEuler, y_InverseEuler, label='Inverse Euler')
plt.plot(x_LeapFrog, y_LeapFrog, label='Leap Frog')
plt.scatter(x_Euler[0], y_Euler[0], color='red', s=100, zorder=5, label='Initial position')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Oscillator')
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
ax.plot(x_LeapFrog, y_LeapFrog, t, label='Leap Frog')
ax.scatter(x_Euler[0], y_Euler[0], t[0], color='red', s=100, label='Initial position')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('t')
ax.set_title('Linear Oscillator')
ax.legend()
plt.show()

"""
----------------------------
STABILITY REGIONS PLOT
----------------------------
"""
z_values = z 
Z_real = real(z_values)
Z_imag = imag(z_values)

def R_euler(z):
    return abs(1 + z)
def R_rk4(z):
    return abs(1 + z + z**2/2 + z**3/6 + z**4/24)
def R_inverse_euler_boundary(z):
    return abs(1 - z)

x = linspace(-4, 2, 400)
y = linspace(-3, 3, 400)
X, Y = meshgrid(x, y)
Z = X + 1j * Y

Mag_Euler = R_euler(Z)
Mag_RK4 = R_rk4(Z)
Mag_InvEuler_Boundary = R_inverse_euler_boundary(Z)

fig, ax = plt.subplots(figsize=(10, 8))
ax.contour(X, Y, Mag_InvEuler_Boundary, levels=[1], colors='green', linewidths=2, linestyles='--')
ax.plot([], [], color='green', linestyle='--', label='Inverse Euler Boundary $|1-z|=1$ (Region is OUTSIDE)')
ax.contour(X, Y, Mag_Euler, levels=[1], colors='blue', linewidths=2)
ax.plot([], [], color='blue', label='Euler Region $|1+z|\leq 1$')
ax.contour(X, Y, Mag_RK4, levels=[1], colors='red', linewidths=2)
ax.plot([], [], color='red', label='RK4 Region $|R(z)|\leq 1$')
ax.fill_between(x, -3, 3, where=x <= 0, color='lightgray', alpha=0.5, label='Cranck-Nicolson Region (LHP)')
ax.plot([0, 0], [-1, 1], color='purple', linewidth=4, label='Leap Frog Region $[-i, i]$')
ax.plot(Z_real, Z_imag, 'o', color='black', markersize=8, label='$z = \lambda \Delta t$ Values')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title('Stability Regions of Temporal Schemes ($z = \lambda \Delta t$)', fontsize=14)
ax.set_xlabel('Real$(z)$', fontsize=12)
ax.set_ylabel('Imaginary$(z)$', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(loc='lower left', fontsize=10)
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(-4, 2)
ax.set_ylim(-3, 3)
plt.show()