"""
Numerical solvers for the Fokker-Planck equations
=================================================

.. currentmodule:: stochpy.fokkerplanck

This module contains numerical solvers for the Fokker-Planck equations associated to diffusion
processes.

For now, it only contains a basic finite difference solver for the 1D case.

.. autoclass:: FokkerPlanck1D
   :members:

"""
import numpy as np
import scipy.integrate as integrate
import scipy.sparse as sps
from . import edpy

class FokkerPlanck1DAbstract:
    """
    Abstract class for 1D Fokker-Planck equations solvers.

    Parameters
    ----------
    drift : function with two variables
        The drift coefficient :math:`a(x, t)`.
    diffusion : function with two variables
        The diffusion coefficient :math:`D(x, t)`.
    """
    def __init__(self, drift, diffusion):
        """
        drift: function of two variables (x, t)
        diffusion: function of two variables (x, t).
        """
        self.drift = drift
        self.diffusion = diffusion


    @classmethod
    def gaussian1d(cls, mean, std, X):
        """
        Return a 1D Gaussian pdf.

        Parameters
        ----------
        mean : float
        std : float
        X : ndarray
            The sample points.

        Returns
        -------
        pdf : ndarray
            The Gaussian pdf at the sample points.
        """
        pdf = np.exp(-0.5*((X-mean)/std)**2)/(np.sqrt(2*np.pi)*std)
        pdf /= integrate.trapz(pdf, X)
        return pdf

    @classmethod
    def dirac1d(cls, pos, X):
        """
        Return a PDF for a certain event.

        Parameters
        ----------
        pos : float
            The value occurring with probability one.
        X : ndarray
            The sample points.

        Returns
        -------
        pdf : ndarray
            The pdf at the sample points.

        Notes
        -----
        The method actually returns a vector with a one at the first sample point larger than `pos`.
        """
        pdf = np.zeros_like(X)
        np.put(pdf, len(X[X < pos]), 1.0)
        pdf /= integrate.trapz(pdf, X)
        return pdf

    @classmethod
    def uniform1d(cls, X):
        """
        Return a uniform PDF.

        Parameters
        ----------
        X : ndarray
            The sample points.

        Returns
        -------
        pdf : ndarray
            The pdf at the sample points.
        """
        pdf = np.ones_like(X)
        pdf /= integrate.trapz(pdf, X)
        return pdf

    def _fpeq(self, P, X, t):
        """
        The equation to solve, to be implemented by the subclass.
        """
        raise NotImplementedError

    def _fpmat(self, X, t):
        """
        The sparse matrix representation of the equation to solve,
        to be implemented by the subclass.
        """
        raise NotImplementedError

    def _fpbc(self, fdgrid, **kwargs):
        """
        Build the boundary condition. To be implemented by the subclass.
        """
        raise NotImplementedError

    def fpintegrate(self, t0, T, **kwargs):
        """
        Numerical integration of the associated Fokker-Planck equation, or its adjoint.

        Parameters
        ----------
        t0 : float
            Initial time.
        T : float
            Integration time.

        Keyword Arguments
        -----------------
        bounds : float 2-tuple
            Domain where we should solve the equation (default (-10.0, 10.0))
        npts : ints
            Number of discretization points in the domain (i.e. spatial resolution). Default: 100.
        dt : float
            Timestep (default choice suitable for the heat equation with forward scheme)
        bc: tuple
            Boundary conditions (a tuple sent to _fpbc)
        method : str
            Numerical scheme: explicit ('euler', default), implicit, or crank-nicolson
        P0 : ndarray
            Initial condition (default is a standard normal distribution).

        Returns
        -------
        t, X, P : float, ndarray, ndarray
            Final time, sample points and solution of the Fokker-Planck
            equation at the sample points.
        """
        # Get computational parameters:
        B, A = kwargs.pop('bounds', (-10.0, 10.0))
        Np = kwargs.pop('npts', 100)
        fdgrid = edpy.RegularCenteredFD(B, A, Np)
        dt = kwargs.pop('dt', 0.25*(np.abs(B-A)/(Np-1))**2/self.diffusion(0.5*(A+B), t0))
        bc = self._fpbc(fdgrid, bc=kwargs.get('bc', ('absorbing', 'absorbing')))
        method = kwargs.pop('method', 'euler')
        # Prepare initial P(x):
        P0 = kwargs.pop('P0', self.gaussian1d(0.0, 1.0, fdgrid.grid))
        # Numerical integration:
        if T > 0:
            if method in ('impl', 'implicit', 'bwd', 'backward',
                          'cn', 'cranknicolson', 'crank-nicolson'):
                return edpy.EDPLinSolver().edp_int(self._fpmat, fdgrid, P0, t0, T, dt, bc,
                                                   scheme=method)
            else:
                return edpy.EDPSolver().edp_int(self._fpeq, fdgrid, P0, t0, T, dt, bc)
        else:
            return t0, fdgrid.grid, P0

    def fpintegrate_generator(self, *args, **kwargs):
        """
        Numerical integration of the associated Fokker-Planck equation, generator version.

        Parameters
        ----------
        *args : variable length argument list
            Times at which to yield the pdf.

        Yields
        ------
        t, X, P : float, ndarray, ndarray
            Time, sample points and solution of the Fokker-Planck equation at the sample points.
        """
        if args:
            t0 = kwargs.pop('t0', args[0])
        for t in args:
            t, X, P = self.fpintegrate(t0, t-t0, **kwargs)
            t0 = t
            kwargs['P0'] = P
            yield t, X, P


class FokkerPlanck1D(FokkerPlanck1DAbstract):
    r"""
    Solver for the 1D Fokker-Planck equation.

    :math:`\partial_t P(x, t) = - \partial_x a(x, t)P(x, t) + \partial^2_{xx} D(x, t) P(x, t)`

    Parameters
    ----------
    drift : function with two variables
        The drift coefficient :math:`a(x, t)`.
    diffusion : function with two variables
        The diffusion coefficient :math:`D(x, t)`.

    Notes
    -----
    This is just the legacy code which was migrated from the
    :class:`stochpy.dynamics.DiffusionProcess1D` class.
    It should be rewritten with a better structure.
    """

    @classmethod
    def from_sde(cls, model):
        r"""
        Construct and return a Fokker-Planck object from a DiffusionProcess object.
        The only thing this constructor does is define the diffusion coefficient :math:`D(x, t)`
        from the diffusion of the stochastic process :math:`\sigma(x, t)` as
        :math:`D(x, t)=\sigma(x, t)^2/2`.
        """
        return FokkerPlanck1D(model.drift, lambda x, t: 0.5*model.diffusion(x, t)**2)

    def _fpeq(self, P, X, t):
        """ Right hand side of the Fokker-Planck equation """
        driftvec = np.array([self.drift(x, t) for x in X.grid])
        diffvec = np.array([self.diffusion(x, t) for x in X.grid])
        return -X.grad(driftvec*P) + X.laplacian(diffvec*P)

    def _fpmat(self, X, t):
        """
        Sparse matrix representation of the linear operator
        corresponding to the RHS of the FP equation
        """
        driftvec = np.array([self.drift(x, t) for x in X.grid])
        diffvec = np.array([self.diffusion(x, t) for x in X.grid])
        Ldrift = -X.grad_mat()*sps.dia_matrix((driftvec, np.array([0])), shape=(X.N, X.N))
        Ldiff = X.lapl_mat()*sps.dia_matrix((diffvec, np.array([0])), shape=(X.N, X.N))
        return Ldrift + Ldiff

    def _fpbc(self, fdgrid, bc=('absorbing', 'absorbing')):
        """ Build the boundary conditions for the Fokker-Planck equation and return it.
        This is useful when at least one of the sides is a reflecting wall. """
        dx = fdgrid.dx
        def refleft(yr, xl, xr, t):
            return yr*self.diffusion(xr, t)/(self.diffusion(xl, t)+self.drift(xl, t)*dx)
        def refright(yl, xl, xr, t):
            return yl*self.diffusion(xl, t)/(self.diffusion(xr, t)-self.drift(xr, t)*dx)
        dic = {('absorbing', 'absorbing'):
                   edpy.DirichletBC([0, 0]),
               ('absorbing', 'reflecting'):
                   edpy.BoundaryCondition(lambda Y, X, t: [0, refright(Y[-2], X[-2], X[-1], t)]),
               ('reflecting', 'absorbing'):
                   edpy.BoundaryCondition(lambda Y, X, t: [refleft(Y[1], X[0], X[1], t), 0]),
               ('reflecting', 'reflecting'):
                   edpy.BoundaryCondition(lambda Y, X, t: [refleft(Y[1], X[0], X[1], t),
                                                           refright(Y[-2], X[-2], X[-1], t)])}
        if bc not in dic:
            raise NotImplementedError("Unknown boundary conditions for the Fokker-Planck equations")
        return edpy.DirichletBC([0, 0]) if self.diffusion == 0 else dic[bc]


class FokkerPlanck1DBackward(FokkerPlanck1DAbstract):
    r"""
    Solver for the adjoint Fokker-Planck equation.

    :math:`\partial_t P(x, t) = a(x, t)\partial_x P(x, t) + D(x, t) \partial^2_{xx} P(x, t)`

    Parameters
    ----------
    drift : function with two variables
        The drift coefficient :math:`a(x, t)`.
    diffusion : function with two variables
        The diffusion coefficient :math:`D(x, t)`.
    """

    def _fpeq(self, P, X, t):
        """
        The adjoint of the Fokker-Planck operator, useful for instance
        in first passage time problems for homogeneous processes.
        """
        driftvec = np.array([self.drift(x, t) for x in X.grid])
        diffvec = np.array([self.diffusion(x, t) for x in X.grid])
        return driftvec[1:-1]*X.grad(P)+diffvec[1:-1]*X.laplacian(P)

    def _fpmat(self, X, t):
        """ Sparse matrix representation of the adjoint of the FP operator """
        driftvec = np.array([self.drift(x, t) for x in X.grid])
        diffvec = np.array([self.diffusion(x, t) for x in X.grid])
        Ladv = sps.dia_matrix((driftvec[1:-1], np.array([0])), shape=(X.N-2, X.N-2))*X.grad_mat()
        Ldiff = sps.dia_matrix((diffvec[1:-1], np.array([0])), shape=(X.N-2, X.N-2))*X.lapl_mat()
        return Ladv + Ldiff

    def _fpbc(self, fdgrid, bc=('absorbing', 'absorbing')):
        """ Build the boundary conditions for the Fokker-Planck equation and return it.
        This is useful when at least one of the sides is a reflecting wall. """
        dic = {('absorbing', 'absorbing'): edpy.DirichletBC([0, 0]),
               #('absorbing', 'reflecting'): edpy.BoundaryCondition(lambda Y, X, t: [0, Y[-2]]),
               ('reflecting', 'absorbing'): edpy.BoundaryCondition(lambda Y, X, t: [Y[1], 0]),
               #('reflecting', 'reflecting'): edpy.BoundaryCondition(lambda Y, X, t: [Y[1], Y[-2]])
               }
        if bc not in dic:
            raise NotImplementedError("Unknown boundary conditions for the Fokker-Planck equations")
        return edpy.DirichletBC([0, 0]) if self.diffusion == 0 else dic[bc]
