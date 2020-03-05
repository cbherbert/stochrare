import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps

###
#   Finite Difference methods
###

class FiniteDifferences:
    """ A simple class to implement finite-difference methods (1D only for now).
    The basic class is just an arbitrary grid.
    Subclasses should provide implementations for things like gradient, laplacian and so on
    (using different schemes), which are the building blocks of the differential operators
    appearing in the equations we want to solve. """

    def __init__(self, X):
        self.grid = X
        self.N = len(X)

class CenteredFD(FiniteDifferences):
    """ Centered finite-difference methods (1D only) """

    def grad(self, Y):
        """ Gradient of a scalar field Y evaluated with a centered finite difference scheme """
        return (Y[2:]-Y[:-2])/(self.grid[2:]-self.grid[:-2])

class RegularCenteredFD(CenteredFD):
    """ Centered finite-difference methods on a regular grid (1D only) """

    def __init__(self, A, B, Np):
        super(self.__class__, self).__init__(np.linspace(A, B, num=Np))
        self.dx = np.abs(A-B)/(Np-1)

    def grad(self, Y):
        """ Gradient of a scalar field Y evaluated with a centered finite difference scheme """
        return (Y[2:]-Y[:-2])/(2*self.dx)

    def laplacian(self, Y):
        """ Laplacian of a scalar field Y evaluated with a centered finite difference scheme """
        return (Y[:-2]+Y[2:]-2*Y[1:-1])/(self.dx**2)

    def grad_mat(self):
        """ Sparse matrix representation of the gradient operator on the grid """
        return sps.dia_matrix((np.array([self.N*[-1.0/(2.0*self.dx)], self.N*[1.0/(2.0*self.dx)]]), np.array([0,2])), shape=(self.N-2, self.N))

    def lapl_mat(self):
        """ Sparse matrix representation of the laplacian operator on the grid """
        return sps.dia_matrix((np.array([self.N*[1.0/(self.dx**2)], self.N*[-2.0/(self.dx**2)], self.N*[1.0/(self.dx**2)]]), np.array([0,1,2])), shape=(self.N-2, self.N))


class RegularForwardFD(FiniteDifferences):
    """ Forward finite-difference methods on a regular grid (1D only) """

    def __init__(self, A, B, Np):
        super(self.__class__, self).__init__(np.linspace(A, B, num=Np))
        self.dx = np.abs(A-B)/(Np-1)

    def grad(self, Y):
        return (Y[1:]-Y[:-1])/(self.dx)

###
#   Boundary Conditions for EDPs
###

class BoundaryCondition:
    """ A generic class for implementing boundary conditions (1D only for now) """
    def __init__(self, fun):
        self.getbc = fun

    def apply(self, Y, X, t):
        Y[(0, -1), ] = self.getbc(Y, X, t)

class DirichletBC(BoundaryCondition):
    """ Dirichlet Boundary Conditions (1D only for now) """
    def __init__(self, Y0):
        super(self.__class__, self).__init__(lambda Y, X, t: Y0)

class NeumannBC(BoundaryCondition):
    """ Neumann Boundary Conditions (1D only for now) """
    def __init__(self, DY0):
        super(self.__class__, self).__init__(lambda Y, X, t: Y[(1, -2), ]-(X[1:]-X[:-1])[(0, -1), ]*DY0)

# class AbsorbingBC(DirichletBC):
#     def __init__(self):
#         super(self.__class__,self).__init__([0,0])


###
#   EDP Solvers
###

class EDPSolver:
    """ Finite-difference solver for partial differential equations """

    @classmethod
    def edp_int(cls, solfun, solgrid, P0, t0, T, dt, bc, **kwargs):
        """ Explicit in time (Euler method) integration of the PDE """
        # time integration
        t = t0
        P = np.copy(P0)
        while (t+dt <= t0+T):
            P[1:-1] += solfun(P, solgrid, t)*dt # Advancing in the bulk
            bc.apply(P, solgrid.grid, t)        # Applying boundary conditions
            t += dt

        return t, solgrid.grid, P

class EDPLinSolver:
    """ Finite-difference solver for linear PDEs.
    For the forward (explicit in time) scheme, this implementation is SLOWER than EDPSolver.
    This class allows for easier implementation of the backward (implicit)
    and Crank-Nicolson schemes. """

    @classmethod
    def edp_int(cls, mat, solgrid, P0, t0, T, dt, bc, **kwargs):
        """ Integration of the PDE, using the scheme given by optional argument 'scheme':
        - expl: forward (Euler) method
        - impl: backward method
        - cn:   Crank-Nicolson method """
        t = t0
        P = np.copy(P0)
        method = kwargs.get('scheme', 'expl')
        while (t+dt <= t0+T):
            if method in ('expl', 'explicit', 'fwd', 'forward'):
                P[1:-1] += dt*mat(solgrid, t).dot(P)
            elif method in ('impl', 'implicit', 'bwd', 'backward'):
                P = sps.linalg.spsolve(sps.eye(solgrid.N)-dt*sps.vstack([sps.coo_matrix((1, solgrid.N)), mat(solgrid, t), sps.coo_matrix((1, solgrid.N))]), P)
            elif method in ('cn', 'cranknicolson', 'crank-nicolson'):
                Lbulk = 0.5*dt*sps.vstack([sps.coo_matrix((1, solgrid.N)),mat(solgrid, t),sps.coo_matrix((1, solgrid.N))])
                P = sps.linalg.spsolve(sps.eye(solgrid.N)-Lbulk, P+Lbulk.dot(P))
            else:
                raise NotImplementedError('The numerical scheme you asked for is not implemented')
            bc.apply(P, solgrid.grid, t)
            t += dt

        return t, solgrid.grid, P
