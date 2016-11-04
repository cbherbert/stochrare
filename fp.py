import numpy as np
import matplotlib.pyplot as plt

# parameters

# model:
# eps = 0.1
# t0 = -10.0
# T = 10.0
# M = 20.0
def a(x,t):
    return x**2+t

# finite differences:
# np = 1000
# B = t0
# dt = 0.01

def grad_centered(Y,dx):
    return (Y[2:]-Y[:-2])/(2*dx)

def grad_forward(Y,dx):
    return (Y[1:]-Y[:-1])/dx

def laplacian_centered(X,dx):
    return (Y[:-2]+Y[2:]-2*Y[1:-1])/(dx**2)



def fpintegrate(t0,T,B,M,Np,eps,dt,**kwargs):

    dx = np.abs(B-M)/Np
    
    # initial P(x)    
    X = np.linspace(B,M,num=Np)
    P = kwargs.get('P',np.exp(-0.5*(X+np.sqrt(np.abs(t0)))**2)/np.sqrt(2*np.pi))    

    # time integration
    t = t0
    while (t < t0+T):
        # Advancing in the bulk:
        deltaP = laplacian_centered(P,dx)
        #drift = grad_centered(np.array([a(x,t) for x in X])*P,dx)
        drift =0.0        
        P[1:-1] += (-drift + eps*deltaP)*dt

        # Absorbing BC at x=M:
        P[-1] = 0
        # # Reflecting BC at x=B:
        # P[0] = P[1]/(1+a(X[0],t)*dx/eps)
        P[0] = 0
    
        t += dt
    return t,P

def testfp(t0,B,M,Np,eps,dt):
    fig = plt.figure()
    ax = plt.axes()

    X = np.linspace(B,M,num=Np)
    #P = np.exp(-0.5*(X+np.sqrt(np.abs(t0)))**2)/np.sqrt(2*np.pi)
    P = np.exp(-0.5*(X-5)**2)/np.sqrt(2*np.pi)
    ax.plot(X,P,label='t='+str(t0))

    t = t0
    for k in xrange(5):
        t,P = fpintegrate(t,100.0,B,M,Np,eps,dt,P=P)
        ax.plot(X,P,label='t='+str(t))

    ax.grid()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$P(x)$')
    #plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
    plt.legend()
    plt.show()
        
