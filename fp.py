import numpy as np
import matplotlib.pyplot as plt

###
#   Finite Difference methods
###

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

###
#   Boundary Conditions for EDPs
###
    
class BoundaryCondition(object):
    """ A generic class for implementing boundary conditions (1D only for now) """
    def __init__(self,fun):
        self.getbc = fun

    def apply(self,Y,X,t):
        Y[(0,-1),] = self.getbc(Y,X,t)

class DirichletBC(BoundaryCondition):
    """ Dirichlet Boundary Conditions (1D only for now) """
    def __init__(self,Y0):
        super(self.__class__,self).__init__(lambda Y,X,t: Y0)

# class AbsorbingBC(DirichletBC):
#     def __init__(self):
#         super(self.__class__,self).__init__([0,0])


###
#   EDP Solvers
###
        
class EDPSolver(object):
    """ Finite-difference solver for partial differential equations """

    def edp_int(self,solfun,solgrid,P0,t0,T,dt,bc):
        """ Explicit in time (Euler method) integration of the PDE """
        # time integration
        t = t0
        P = P0
        while (t < t0+T):            
            P[1:-1] += solfun(P,solgrid,t)*dt # Advancing in the bulk
            bc.apply(P,solgrid.grid,t)        # Applying boundary conditions
            t += dt
                
        return t,P

    
###
#   Test: numerical integration of the heat equation
###

def testfp(t0,B,M,Np,dt,niter,Deltat,**kwargs):

    fig = plt.figure()
    ax = plt.axes()

    X = np.linspace(B,M,num=Np)
    P = kwargs.get('P',np.exp(-0.5*X**2)/np.sqrt(2*np.pi))    
    model = kwargs.get('model',Wiener())

    ax.plot(X,P,label='t='+str(t0))
    t = t0
    for k in xrange(niter):
        t,P = model.fpintegrate(t,Deltat,B,M,Np,dt,P=P)
        ax.plot(X,P,label='t='+str(t))

    ax.grid()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$P(x,t)$')
    #plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
    plt.legend()
    plt.show()

def testfp_diffusion(t0,Np,dt):
    testfp(t0,-10.0,10.0,Np,dt,5,1.0)

def testfp_saddlenode(t0,M,Np,dt):
    testfp(t0,t0,M,Np,dt,2,1.0,model=StochSaddleNode(1.0),P=np.exp(-0.5*(np.linspace(t0,M,num=Np)+np.sqrt(np.abs(t0)))**2)/np.sqrt(2*np.pi))
    
