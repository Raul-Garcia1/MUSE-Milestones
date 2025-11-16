"""
Ampliación de Matemáticas - Master Universitario en Sistemas Espaciales - ETSIAE
Milestone 6: Lagrange points and their stability.
 1. Write a high order embedded Runge-Kutta method.
 2. Write function to simulate the circular restricted three body problem.
 3. Determination of the Lagrange points F(U) = 0.
 4. Stability of the Lagrange points: L1,L2,L3,L4,L5.
 5. Orbits around the Lagrange points by means of different temporal schemes.
"""

from numpy import array,concatenate,zeros,abs,max,log,real,imag,isclose,eye,block,all,meshgrid,linspace,finfo,copy,sqrt
from numpy.linalg import norm,eigvals
from scipy.optimize import fsolve
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

def N_Body_Problem_2D(dr,r): 
    N=len(r)//2
    F1=dr
    F2=zeros((N*2))
    for i in range(N):
        i_idx = 2 * i
        r_i = r[i_idx:i_idx + 2] # Extract position vector [x_i, y_i]
        for j in range(N):
            if i != j:
                j_idx = 2 * j
                r_j = r[j_idx:j_idx + 2] # Extract position vector [x_j, y_j]
                F2[i_idx:i_idx + 2]= F2[i_idx:i_idx + 2] +(r_j-r_i)/norm(r_j-r_i)**3

    F_resultante=concatenate((F1,F2))
    
    return F_resultante

def N_Body_Problem_3D(dr,r): 
    N=len(r)//3
    F1=dr
    F2=zeros((N*3))
    for i in range(N):
        i_idx = 3 * i
        r_i = r[i_idx:i_idx + 3] # Extract position vector [x_i, y_i]
        for j in range(N):
            if i != j:
                j_idx = 3 * j
                r_j = r[j_idx:j_idx + 3] # Extract position vector [x_j, y_j]
                F2[i_idx:i_idx + 3]= F2[i_idx:i_idx + 3] +(r_j-r_i)/norm(r_j-r_i)**3
    
    F_resultante=concatenate((F1,F2))

    return F_resultante

"""
----------------------------
1. EMBEDDED RUNGE-KUTTA SCHEME RK45
----------------------------
EMBEDDED RUNGE-KUTTA SCHEME RK45 with adaptative time step
        k1 = delta_T*F(t_n, dr_n, r_n)
        k2 = delta_T*F(t_n + delta_T/4, dr_n + 1/4*k1, r_n + 1/4*k1)
        k3 = delta_T*F(t_n + delta_T*3/8, dr_n + 3/32*k1 + 9/32*k2, r_n + 3/32*k1 + 9/32*k2)
        k4 = delta_T*F(t_n + delta_T*12/13, dr_n + 1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3, r_n + 1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3)
        k5 = delta_T*F(t_n + delta_T, dr_n + 439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4, r_n + 439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4)
        k6 = delta_T*F(t_n + delta_T/2, dr_n - 8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5, r_n - 8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5)
        Order 4 estimation--> Uo4_n+1 = U_n + (k1*25/216 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)
        Order 5 estimation--> Uo5_n+1 = U_n + (k1*16/135 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)
        Error_array=U_o5_n+1-U_o4_n+1 
        Tol=Tol_a+Tol_r*norm(U_n)
        E = sqrt (1/dim * Sum (Error_array(j)/Tol(j)))
        S=(1/E)^(1/5) where 5 is p+1 of the lower order method
        if E<=1: Integration continues
            U_n+1=Uo5_n+1
            delta_T=delta_T*S
        elif 1<=E: Integration is repeated with a lower step
            delta_T=delta_T*S
            i=i-1

"""
def RK45(F,U0,n,delta_T):
    dim=len(U0)
    U_Runge=zeros((dim+1,n)) #Definition of the size of U state vector where each row is [r(1x2),dr(1x2),t]'
    U_Runge[0:dim,0]= U0 #Include initial conditions in U
    k1=zeros(dim)
    k2=zeros(dim)
    k3=zeros(dim)
    k4=zeros(dim)
    k5=zeros(dim)
    k6=zeros(dim)
    U_orden4=zeros(dim)
    U_orden5=zeros(dim)
    Error_array=zeros(dim)
    Tol=zeros(dim)
    Tol_a=1e-10
    Tol_r=1e-10

    for i in range (1,n):
        E=2
        delta_T_attempt=delta_T
        while E>1:
            k1=delta_T_attempt*F(U_Runge[dim//2:dim,i-1],U_Runge[0:dim//2,i-1]) #k1 = delta_T*F(t_n, dr_n, r_n)
            k2=delta_T_attempt*F(U_Runge[dim//2:dim,i-1] + 1/4*k1[dim//2:dim],U_Runge[0:dim//2,i-1] + 1/4*k1[0:dim//2]) #k2 = delta_T*F(t_n + delta_T/4, dr_n + 1/4*k1, r_n + 1/4*k1) 
            k3=delta_T_attempt*F(U_Runge[dim//2:dim,i-1] + 3/32*k1[dim//2:dim] + 9/32*k2[dim//2:dim],U_Runge[0:dim//2,i-1]+ 3/32*k1[0:dim//2] + 9/32*k2[0:dim//2]) #k3 = delta_T*F(t_n + delta_T*3/8, dr_n + 3/32*k1 + 9/32*k2, r_n + 3/32*k1 + 9/32*k2)
            k4=delta_T_attempt*F(U_Runge[dim//2:dim,i-1] + 1932/2197*k1[dim//2:dim] - 7200/2197*k2[dim//2:dim] + 7296/2197*k3[dim//2:dim],U_Runge[0:dim//2,i-1] + 1932/2197*k1[0:dim//2] - 7200/2197*k2[0:dim//2] + 7296/2197*k3[0:dim//2]) #k4 = delta_T*F(t_n + delta_T*12/13, dr_n + 1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3, r_n + 1932/2197*k1 - 7200/2197*k2 + 7296/2197*k3)
            k5=delta_T_attempt*F(U_Runge[dim//2:dim,i-1] + 439/216*k1[dim//2:dim] - 8*k2[dim//2:dim] + 3680/513*k3[dim//2:dim] - 845/4104*k4[dim//2:dim],U_Runge[0:dim//2,i-1] + 439/216*k1[0:dim//2] - 8*k2[0:dim//2] + 3680/513*k3[0:dim//2] - 845/4104*k4[0:dim//2]) #k5 = delta_T*F(t_n + delta_T, dr_n + 439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4, r_n + 439/216*k1 - 8*k2 + 3680/513*k3 - 845/4104*k4)
            k6=delta_T_attempt*F(U_Runge[dim//2:dim,i-1] - 8/27*k1[dim//2:dim] + 2*k2[dim//2:dim] - 3544/2565*k3[dim//2:dim] + 1859/4104*k4[dim//2:dim] - 11/40*k5[dim//2:dim],U_Runge[0:dim//2,i-1] - 8/27*k1[0:dim//2] + 2*k2[0:dim//2] - 3544/2565*k3[0:dim//2] + 1859/4104*k4[0:dim//2] - 11/40*k5[0:dim//2]) #k6 = delta_T*F(t_n + delta_T/2, dr_n - 8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5, r_n - 8/27*k1 + 2*k2 - 3544/2565*k3 + 1859/4104*k4 - 11/40*k5)
            U_orden4= U_Runge[0:dim,i-1] + (k1*25/216 + 0*k2 +  1408/2565*k3 + 2197/4104*k4 - 1/5*k5) #Order 4 estimation--> Uo4_n+1 = U_n + (k1*25/216 + 1408/2565*k3 + 2197/4104*k4 - 1/5*k5)
            U_orden5= U_Runge[0:dim,i-1] + (k1*16/135 + 0*k2 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6) #Order 5 estimation--> Uo5_n+1 = U_n + (k1*16/135 + 6656/12825*k3 + 28561/56430*k4 - 9/50*k5 + 2/55*k6)
            Error_array=U_orden5-U_orden4 
            Tol=Tol_a+Tol_r*abs(U_Runge[0:dim,i-1])
            Ej_Tolj=0
            for j in range(dim):
                Ej_Tolj=Ej_Tolj + (Error_array[j]/Tol[j])**2 # Sum (Error_array(j)/Tol(j))
                E=sqrt(1/dim*Ej_Tolj) # E = sqrt (1/dim * Sum (Error_array(j)/Tol(j)))
            S=(1/E)**(1/5) # S=(1/E)^(1/5) where 5 is p+1 of the lower order method
            if E>1:
                delta_T_attempt=delta_T_attempt*S
        U_Runge[0:dim,i]=U_orden5
        U_Runge[dim,i]=U_Runge[dim,i-1]+delta_T_attempt #Include time in U[5,:]
        delta_T=delta_T_attempt*S
    return U_Runge

"""
----------------------------
2. CIRCULAR RESTRICTED 3 BODY PROBLEM
----------------------------
General N body problem
d2r_i/dt = - Sum(G * m_i * (r_j - r_i) / |r_j - r_i|^3)
 Sum for j inequal i to N

2 big masses M1,M2 and the third one is very small m3=0. With this assumption, M1,M2 will have a Keplerian 2 body orbit

Adimensional units: - M1+M2=1
                    - Distance between M1 and M2 is R=1
                    - Gravitatory Constant G=1
                    - Angular velocity constant (for circular orbit) w=1

Mass parameter mu=M2/(M1+M2); M1=1-mu; M2=mu

Reference system rotates with M1,M2 with w. M1 and M2 are fixed in this reference system. M1(-mu,0), M2(1-mu,0). The origin is the CM

The problem will study the movement of m3. The equation:

d2r/dt = - grad(pot) - 2w x dr/dt where - 2w x dr/dt is the Coriolis Force. Fx (circular) = 2*dy/dt; Fy (circular) = -2*dx/dt
and pot= pot_grav + pot_centrif = -GM1/sqrt((x-x1)^2+y^2) -GM2/sqrt((x-x2)^2+y^2) -1/2(x^2+y^2)= -(1-mu)/sqrt((x+mu)^2+y^2) -mu/sqrt((x-1+mu)^2+y^2) -1/2(x^2+y^2)

Then d2x/dt= 2*dy/dt - d(pot)/dx = 2*dy/dt + x -(1-mu)*(x+mu)/sqrt((x+mu)^2+y^2)^3 -mu*(x-1+mu)/sqrt((x-1+mu)^2+y^2)^3 
     d2y/dt= -2*dx/dt - d(pot)/dx = -2*dx/dt + y -(1-mu)*y/sqrt((x+mu)^2+y^2)^3 -mu*y/sqrt((x-1+mu)^2+y^2)^3
"""
def Circular_Restricted_3_Body_Problem_2D(dr,r): 
    mu=0.01215058 #Earth-Moon
    F1=dr
    F2=zeros((2))
    F2[0]=2*dr[1] + r[0] -(1-mu)*(r[0]+mu)/sqrt((r[0]+mu)**2+r[1]**2)**3 -mu*(r[0]-1+mu)/sqrt((r[0]-1+mu)**2+r[1]**2)**3 
    F2[1]=-2*dr[0] + r[1] -(1-mu)*r[1]/sqrt((r[0]+mu)**2+r[1]**2)**3 -mu*r[1]/sqrt((r[0]-1+mu)**2+r[1]**2)**3
    

    F_resultante=concatenate((F1,F2))
    
    return F_resultante

"""
----------------------------
2. LAGRANGE POINTS FOR CIRCULAR RESTRICTED 3 BODY PROBLEM
----------------------------
d2x/dt=dx/dt=0
d2x/dt=dx/dt=0

Then: x -(1-mu)*(x+mu)/sqrt((x+mu)^2+y^2)^3 -mu*(x-1+mu)/sqrt((x-1+mu)^2+y^2)^3=0
      y -(1-mu)*y/sqrt((x+mu)^2+y^2)^3 -mu*y/sqrt((x-1+mu)^2+y^2)^3=0
"""
def Lagrange_Points(mu):
    
    def Equation(x):
        return x - (1 - mu) * (x + mu) / (x + mu)**3 - mu * (x - 1 + mu) / (x - 1 + mu)**3
    x_L1_guess = 1 - mu - 0.1 
    L1_x = fsolve(Equation, x_L1_guess)[0]
    x_L2_guess = 1 - mu + 0.1 
    L2_x = fsolve(Equation, x_L2_guess)[0]
    x_L3_guess = -mu - 1.05 
    L3_x = fsolve(Equation, x_L3_guess)[0]
    
    x_L45 = 0.5 - mu
    y_L4 = sqrt(3) / 2
    y_L5 = -sqrt(3) / 2
    
    L_points = {
        'L1': (L1_x, 0.0),
        'L2': (L2_x, 0.0),
        'L3': (L3_x, 0.0),
        'L4': (x_L45, y_L4),
        'L5': (x_L45, y_L5),
    }
    
    return L_points



r0 = array([0.487,0.86]) #Definition of initial position
dr0 = array([0, 0]) #Definition of initial velocity
delta_T=0.001 #Definition of Delta T = t_total/n
n=10000 #Definition of n number of steps
U0=concatenate((r0,dr0)) #initial conditions U0 
mu=0.01215058

U_Runge=Cauchy(RK45,Circular_Restricted_3_Body_Problem_2D,U0,n,delta_T)

L_points=Lagrange_Points(mu)
L1_x, L1_y = L_points['L1']
L2_x, L2_y = L_points['L2']
L3_x, L3_y = L_points['L3']
L4_x, L4_y = L_points['L4']
L5_x, L5_y = L_points['L5']

print(f"L1: x = {L1_x:.6f}, y = 0.000000")
print(f"L2: x = {L2_x:.6f}, y = 0.000000")
print(f"L3: x = {L3_x:.6f}, y = 0.000000")
print(f"L4: x = {L4_x:.6f}, y = {L4_y:.6f}")
print(f"L5: x = {L5_x:.6f}, y = {L5_y:.6f}")

"""
----------------------------
2D AND 3D GRAPHS
----------------------------
"""

# --- 2D GRAPH ---


x_3 = U_Runge[0, :]
y_3 = U_Runge[1, :]


plt.figure(figsize=(8, 8))
plt.plot(x_3, y_3, label='Body 3')
plt.scatter(-mu,0, color='blue', s=100, zorder=5, label='Initial position Body 1 (Earth)')
plt.scatter(1-mu,0, color='black', s=100, zorder=5, label='Initial position Body 2 (Moon)')
plt.scatter(L1_x, 0, color='green', marker='D', s=50, zorder=6, label='$L_1$')
plt.scatter(L2_x, 0, color='green', marker='D', s=50, zorder=6, label='$L_2$')
plt.scatter(L3_x, 0, color='green', marker='D', s=50, zorder=6, label='$L_3$')
plt.scatter(L4_x, L4_y, color='purple', marker='*', s=150, zorder=6, label='$L_4$')
plt.scatter(L5_x, L5_y, color='purple', marker='*', s=150, zorder=6, label='$L_5$')
plt.scatter(x_3[0], y_3[0], color='red', s=20, zorder=5, label='Initial position Body 3')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('N Body Problem')
plt.grid(True) 
plt.legend()  
max_abs_coord = max([abs(L3_x), abs(L4_y), abs(L5_y)])
plot_limit = max_abs_coord * 1.1 
plt.xlim(-plot_limit, plot_limit) 
plt.ylim(-plot_limit, plot_limit)
plt.axis('equal') 
plt.show() 


