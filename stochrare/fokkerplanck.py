"""
Numerical solvers for the Fokker-Planck equations
=================================================

.. currentmodule:: stochrare.fokkerplanck

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


class FokkerPlanck1D:
    r"""
    Solver for the 1D Fokker-Planck equation.

    :math:`\partial_t P(x,t) = - \partial_x a(x,t)P(x,t) + D \partial^2_{xx} P(x,t)`

    Parameters
    ----------
    drift : function with two variables
        The drift coefficient :math:`a(x, t)`.
    diffusion : float
        The constant diffusion coefficient :math:`D`.

    Notes
    -----
    This is just the legacy code which was migrated from the
    :class:`stochrare.dynamics.DiffusionProcess1D` class.
    It should be rewritten with a better structure.
    In particular, it only works with a constant diffusion for now.
    """
    def __init__(self, drift, diffusion):
        """
        drift: function of two variables (x, t)
        diffusion: diffusion coefficient (float).
        """
        self.drift = drift
        self.diffusion = diffusion

    def _fpeq(self, P, X, t):
        """ Right hand side of the Fokker-Planck equation """
        return -X.grad(self.drift(X.grid, t)*P) + self.diffusion*X.laplacian(P)

    def _fpadj(self, G, X, t):
        """
        The adjoint of the Fokker-Planck operator, useful for instance
        in first passage time problems for homogeneous processes.
        """
        return self.drift(X.grid, t)[1:-1]*X.grad(G)+self.diffusion*X.laplacian(G)

    def _fpmat(self, X, t):
        """
        Sparse matrix representation of the linear operator
        corresponding to the RHS of the FP equation
        """
        return -X.grad_mat()*sps.dia_matrix((self.drift(X.grid, t), np.array([0])),
                                            shape=(X.N, X.N)) + self.diffusion*X.lapl_mat()

    def _fpadjmat(self, X, t):
        """ Sparse matrix representation of the adjoint of the FP operator """
        return sps.dia_matrix((self.drift(X.grid, t)[1:-1], np.array([0])),
                              shape=(X.N-2, X.N-2))*X.grad_mat() + self.diffusion*X.lapl_mat()

    def _fpbc(self, fdgrid, bc=('absorbing', 'absorbing'), **kwargs):
        """ Build the boundary conditions for the Fokker-Planck equation and return it.
        This is useful when at least one of the sides is a reflecting wall. """
        dx = fdgrid.dx
        dic = {('absorbing', 'absorbing'): edpy.DirichletBC([0, 0]),
               ('absorbing', 'reflecting'): edpy.BoundaryCondition(lambda Y, X, t: [0,Y[-2]/(1-self.drift(X[-1], t)*dx/self.diffusion)]),
               ('reflecting', 'absorbing'): edpy.BoundaryCondition(lambda Y, X, t: [Y[1]/(1+self.drift(X[0], t)*dx/self.diffusion),0]),
               ('reflecting', 'reflecting'): edpy.BoundaryCondition(lambda Y, X, t: [Y[1]/(1+self.drift(X[0], t)*dx/self.diffusion), Y[-2]/(1-self.drift(X[-1], t)*dx/self.diffusion)])}
        if bc not in dic:
            raise NotImplementedError("Unknown boundary conditions for the Fokker-Planck equations")
        return edpy.DirichletBC([0, 0]) if self.diffusion == 0 else dic[bc]

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
            Domain where we should solve the equation (default (-10.0,10.0))
        npts : ints
            Number of discretization points in the domain (i.e. spatial resolution). Default: 100.
        dt : float
            Timestep (default choice suitable for the heat equation with forward scheme)
        bc: stochrare.edpy.BoundaryCondition object or tuple
            Boundary conditions (either a BoundaryCondition object or a tuple sent to _fpbc)
        method : str
            Numerical scheme: explicit ('euler', default), implicit, or crank-nicolson
        adjoint : bool
            Integrate the adjoint FP rather than the forward FP (default False).
        P0 : str
            Initial condition: 'gauss' (default), 'dirac' or 'uniform'.

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
        dt = kwargs.pop('dt', 0.25*(np.abs(B-A)/(Np-1))**2/self.diffusion)
        bc = self._fpbc(fdgrid, **kwargs)
        method = kwargs.pop('method', 'euler')
        adj = kwargs.pop('adjoint', False)
        # Prepare initial P(x):
        P0 = kwargs.pop('P0', 'gauss')
        if P0 == 'gauss':
            P0 = self.gaussian1d(kwargs.get('P0center', 0.0), kwargs.get('P0std', 1.0), fdgrid.grid)
        if P0 == 'dirac':
            P0 = np.zeros_like(fdgrid.grid)
            np.put(P0, len(fdgrid.grid[fdgrid.grid < kwargs.get('P0center', 0.0)]), 1.0)
            P0 /= integrate.trapz(P0, fdgrid.grid)
        if P0 == 'uniform':
            P0 = np.ones_like(fdgrid.grid)
            P0 /= integrate.trapz(P0, fdgrid.grid)
        # Numerical integration:
        if T > 0:
            if method in ('impl', 'implicit', 'bwd', 'backward',
                          'cn', 'cranknicolson', 'crank-nicolson'):
                fpmat = {False: self._fpmat, True: self._fpadjmat}.get(adj)
                return edpy.EDPLinSolver().edp_int(fpmat, fdgrid, P0, t0, T, dt, bc, scheme=method)
            else:
                fpfun = {False: self._fpeq, True: self._fpadj}.get(adj)
                return edpy.EDPSolver().edp_int(fpfun, fdgrid, P0, t0, T, dt, bc)
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
        fun = kwargs.pop('integ', self.fpintegrate)
        for t in args:
            t, X, P = fun(t0, t-t0, **kwargs)
            t0 = t
            kwargs['P0'] = P
            yield t, X, P
