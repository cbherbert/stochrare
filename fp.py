import numpy as np
import matplotlib.pyplot as plt

# finite differences:

class FiniteDifferences(object):
    """ A simple class to implement finite-difference methods (1D only for now) """

    def __init__(self,X):
        self.grid = X

class CenteredFD(FiniteDifferences):
    """ Centered finite-difference methods (1D only) """

    def grad(self,Y):
        """ Gradient of a scalar field Y evaluated with a centered finite difference scheme """
        return (Y[2:]-Y[:-2])/(self.grid[2:]-self.grid[:-2])

class RegularCenteredFD(CenteredFD):
    """ Centered finite-difference methods on a regular grid (1D only) """
    
    def __init__(self,A,B,Np):
        super(self.__class__,self).__init__(np.linspace(A,B,num=Np))
        self.dx = np.abs(A-B)/(Np-1)

    def grad(self,Y):
        """ Gradient of a scalar field Y evaluated with a centered finite difference scheme """
        return (Y[2:]-Y[:-2])/(2*self.dx)
        
    def laplacian(self,Y):
        """ Laplacian of a scalar field Y evaluated with a centered finite difference scheme """
        return (Y[:-2]+Y[2:]-2*Y[1:-1])/(self.dx**2)
        
class RegularForwardFD(FiniteDifferences):
    """ Forward finite-difference methods on a regular grid (1D only) """

    def __init__(self,A,B,Np):
        super(self.__class__,self).__init__(np.linspace(A,B,num=Np))
        self.dx = np.abs(A-B)/(Np-1)

    def grad(self,Y):
        return (Y[1:]-Y[:-1])/(self.dx)



# Test: numerical integration of the heat equation

def testfp(t0,Np,dt):

    model = Wiener()
    
    fig = plt.figure()
    ax = plt.axes()

    B=-10.0
    M=10.0
    X = np.linspace(B,M,num=Np)

    #P = np.exp(-0.5*(X+np.sqrt(np.abs(t0)))**2)/np.sqrt(2*np.pi)
    t = t0
    P = np.exp(-0.5*X**2)/np.sqrt(2*np.pi)
    ax.plot(X,P,label='t='+str(t0))
    
    for k in xrange(5):
        t,P = model.fpintegrate(t,1.0,B,M,Np,dt,P=P)
        ax.plot(X,P,label='t='+str(t))

    ax.grid()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$P(x,t)$')
    #plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
    plt.legend()
    plt.show()
        
def testfp2(t0,M,Np,dt):

    model = StochSaddleNode(1.0)

    fig = plt.figure()
    ax = plt.axes()

    B = t0
    X = np.linspace(B,M,num=Np)
    P = np.exp(-0.5*(X+np.sqrt(np.abs(t0)))**2)/np.sqrt(2*np.pi)    
    ax.plot(X,P,label='t='+str(t0))

    t = t0
    for k in xrange(2):
        t,P = model.fpintegrate(t,1.0,B,M,Np,dt,P=P)
        ax.plot(X,P,label='t='+str(t))

    ax.grid()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$P(x,t)$')
    #plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
    plt.legend()
    plt.show()
