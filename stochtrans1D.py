import numpy as np
import matplotlib.pyplot as plt
from numba import float32,float64,vectorize,autojit,jit
import scipy.integrate as integrate
import scipy.sparse as sps
from scipy.interpolate import interp1d
from scipy.special import airy, ai_zeros, gamma,gammaincc
from scipy.optimize import brentq
import edpy, data

class StochModel(object):
    """ The generic class from which all the models I consider derive.
        It corresponds to the family of 1D SDEs dx_t = F(x_t,t)dt + sqrt(2*D0)dW_t """

    default_dt = 0.1

    def __init__(self,vecfield,Damp):
        """ vecfield is a function of two variables (x,t) and Damp the amplitude of the diffusion term (noise) """
        self.F  = vecfield
        self.D0 = Damp

    def potential(self,X,t):
        """ Integrate the vector field to obtain the value of the underlying potential at the input points.
        Caveat: This works only because we restrict ourselves to 1D models. """
        fun = interp1d(X,-self.F(X,t))
        return np.array([integrate.quad(fun,0.0,x)[0] for x in X])

    def increment(self,x,t,**kwargs):
        """ Return F(x_t,t)dt + sqrt(2*D0)dW_t """
        dt = kwargs.get('dt',self.default_dt)
        return self.F(x,t) * dt + np.sqrt(2*self.D0*dt)*np.random.normal(0.0,1.0)

    def time_reversal(self):
        """ Apply time reversal and return the new model """
        return StochModel_T(lambda x,t: -self.F(x,-t),self.D0)

    def trajectory(self,x0,t0,**kwargs):
        """ Integrate a trajectory with given initial condition (t0,x0) """
        x      = [x0]
        dt     = kwargs.get('dt',self.default_dt) # Time step
        time   = kwargs.get('T',10)   # Total integration time
        if dt < 0: time=-time
        tarray = np.linspace(t0,t0+time,num=time/dt+1)
        for t in tarray[1:]:
            x += [ x[-1] + self.increment(x[-1],t,dt=dt)]
        x = np.array(x)
        if kwargs.get('finite',False):
            tarray = tarray[np.isfinite(x)]
            x = x[np.isfinite(x)]
        return tarray,x

    def traj_cond_gen(self,x0,t0,tau,M,**kwargs):
        """Generate trajectories conditioned on the first-passage time tau at value M.
        Initial conditions are (x0,t0).
        Optional keyword arguments:
        - dt     -- integration timestep (default is self.default_dt)
        - ttol   -- first-passage time tolerance (default is 1% of trajectory duration)
        - num    -- number of trajectories generated (default is 10)
        - interp -- interpolate to generate unifomly sampled trajectories
        - npts   -- number of points for interpolated trajectories (default (tau-t0)/dt)
        """
        dt      = kwargs.get('dt',self.default_dt)
        tau_tol = kwargs.get('ttol',0.01*np.abs(tau-t0))
        num     = kwargs.pop('num',10)
        interp  = kwargs.pop('interp',False)
        npts    = kwargs.pop('npts',np.abs(tau-t0)/dt)
        while (num>0):
            x = [x0]
            t = [t0]
            while (x[-1]<=M and t[-1]<=tau):
                x += [ x[-1] + self.increment(x[-1],t[-1],dt=dt)]
                t += [ t[-1] + dt ]
            if (x[-1]>M and np.abs(t[-1]-tau)<tau_tol):
                num -= 1
                if interp:
                   fun = interp1d(t,x,fill_value='extrapolate')
                   t = np.linspace(t0,tau,num=npts)
                   x = fun(t)
                yield t,x

    @classmethod
    def traj_fpt(self,M,*args):
        """ Compute the first passage time for each trajectory given as argument """
        for tt,xx in args:
            for t,x in zip(tt,xx):
                if x>M:
                    yield t
                    break

    @classmethod
    def trajectoryplot(cls,*args,**kwargs):
        """ Plot previously computed trajectories """
        fig = plt.figure()
        ax  = plt.axes()
        lines = []
        for t,x in args:
            lines += ax.plot(t,x)

        ax.grid()
        ax.set_ylim(kwargs.get('ylim',ax.get_ylim()))
        ax.set_xlim(kwargs.get('xlim',ax.get_xlim()))
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x(t)$')
        plottitle = kwargs.get('title',"")
        if plottitle != "":
            plt.title(plottitle)

        labels = kwargs.get('labels',[])
        if labels != []:
            plt.legend(lines,labels,bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)

        cls._trajectoryplot_decorate(*args,axis=ax,**kwargs)
        plt.show()

    @classmethod
    def _trajectoryplot_decorate(cls,*args,**kwargs):
        pass

    def blowuptime(self,x0,t0,**kwargs):
        """ Compute the last time with finite values, for one realization"""
        t,x = self.trajectory(x0,t0,**kwargs)
        return t[np.isfinite(x)][-1]

    def _fpeq(self,P,X,t):
        """ Right hand side of the Fokker-Planck equation associated to the stochastic process """
        return -X.grad(self.F(X.grid,t)*P) + self.D0*X.laplacian(P)

    def _fpadj(self,G,X,t):
        """ The adjoint of the Fokker-Planck operator, useful for instance in first passage time problems for homogeneous processes. """
        return self.F(X.grid,t)[1:-1]*X.grad(G)+self.D0*X.laplacian(G)

    def _fpmat(self,X,t):
        """ Sparse matrix representation of the linear operator corresponding to the RHS of the FP equation """
        return -X.grad_mat()*sps.dia_matrix((self.F(X.grid,t),np.array([0])),shape=(X.N,X.N)) + self.D0*X.lapl_mat()

    def _fpadjmat(self,X,t):
        """ Sparse matrix representation of the adjoint of the FP operator """
        return sps.dia_matrix((self.F(X.grid,t)[1:-1],np.array([0])),shape=(X.N-2,X.N-2))*X.grad_mat() + self.D0*X.lapl_mat()

    def _fpbc(self,fdgrid,bc=('absorbing','absorbing'),**kwargs):
        """ Build the boundary conditions for the Fokker-Planck equation and return it.
        This is useful when at least one of the sides is a reflecting wall. """
        dx = fdgrid.dx
        dic = { ('absorbing', 'absorbing'): edpy.DirichletBC([0,0]),
                    ('absorbing','reflecting'): edpy.BoundaryCondition(lambda Y,X,t: [0,Y[-2]/(1-self.F(X[-1],t)*dx/self.D0)]),
                    ('reflecting','absorbing'): edpy.BoundaryCondition(lambda Y,X,t: [Y[1]/(1+self.F(X[0],t)*dx/self.D0),0]),
                    ('reflecting','reflecting'): edpy.BoundaryCondition(lambda Y,X,t: [Y[1]/(1+self.F(X[0],t)*dx/self.D0),Y[-2]/(1-self.F(X[-1],t)*dx/self.D0)])}
        if bc in dic:
            return edpy.DirichletBC([0,0]) if self.D0 == 0 else dic[bc]
        else:
            return bc

    def _fpthsol(self,X,t,**kwargs):
        """ Analytic solution of the Fokker-Planck equation, when it is known.
        In general this is an empty method but subclasses corresponding to stochastic processes
        for which theoretical results exists should override it."""
        pass

    def fpintegrate(self,t0,T,**kwargs):
        """
        Numerical integration of the associated Fokker-Planck equation, or its adjoint.
        Optional arguments are the following:
        - bounds=(-10.0,10.0); domain where we should solve the equation
        - npts=100;            number of discretization points in the domain (i.e. spatial resolution)
        - dt;                  timestep (default choice suitable for the heat equation with forward scheme)
        - bc;                  boundary conditions (either a BoundaryCondition object or a tuple sent to _fpbc)
        - method=euler;        numerical scheme: explicit (default), implicit, or crank-nicolson
        - adj=False;           integrate the adjoint FP rather than the forward FP?
        """
        # Get computational parameters:
        B,A    = kwargs.pop('bounds',(-10.0,10.0))
        Np     = kwargs.pop('npts',100)
        fdgrid = edpy.RegularCenteredFD(B,A,Np)
        dt     = kwargs.pop('dt',0.25*(np.abs(B-A)/(Np-1))**2/self.D0)
        bc     = self._fpbc(fdgrid,**kwargs)
        method = kwargs.pop('method','euler')
        adj    = kwargs.pop('adjoint',False)
        # Prepare initial P(x):
        P0     = kwargs.pop('P0','gauss')
        if P0 is 'gauss':
            P0 = np.exp(-0.5*((fdgrid.grid-kwargs.get('P0center',0.0))/kwargs.get('P0std',1.0))**2)/(np.sqrt(2*np.pi)*kwargs.get('P0std',1.0))
            P0 /= integrate.trapz(P0,fdgrid.grid)
        if P0 is 'dirac':
            P0 = np.zeros_like(fdgrid.grid)
            np.put(P0,len(fdgrid.grid[fdgrid.grid<kwargs.get('P0center',0.0)]),1.0)
            P0 /= integrate.trapz(P0,fdgrid.grid)
        if P0 is 'uniform':
            P0 = np.ones_like(fdgrid.grid)
            P0 /= integrate.trapz(P0,fdgrid.grid)
        # Numerical integration:
        if T>0:
            if method in ('impl','implicit','bwd','backward','cn','cranknicolson','crank-nicolson'):
                fpmat = {False: self._fpmat, True: self._fpadjmat}.get(adj)
                return edpy.EDPLinSolver().edp_int(fpmat,fdgrid,P0,t0,T,dt,bc,scheme=method)
            else:
                fpfun = {False: self._fpeq, True: self._fpadj}.get(adj)
                return edpy.EDPSolver().edp_int(fpfun,fdgrid,P0,t0,T,dt,bc)
        else:
            return t0,fdgrid.grid,P0

    def pdfgen(self,*args,**kwargs):
        """ Generate the pdf solution of the FP equation at various times """
        t0  = kwargs.pop('t0',args[0])
        fun = kwargs.pop('integ',self.fpintegrate)
        for t in args:
            t,X,P = fun(t0,t-t0,**kwargs)
            t0 = t
            kwargs['P0'] = P
            yield t, X, P


    def pdfplot(self,*args,**kwargs):
        """ Plot the pdf P(x,t) at various times """
        fig = plt.figure()
        ax = plt.axes()
        t0  = kwargs.pop('t0',args[0])
        fun = kwargs.pop('integ',self.fpintegrate)
        if kwargs.get('potential',False):
            ax2 = ax.twinx()
            ax2.set_ylabel('$V(x,t)$')
        for t in args:
            t,X,P = fun(t0,t-t0,**kwargs)
            line, = ax.plot(X,P,label='t='+format(t,'.2f'))
            if kwargs.get('th',False):
                Pth = self._fpthsol(X,t,**kwargs)
                if Pth is not None: ax.plot(X,Pth,color=line.get_color(),linestyle='dotted')
            if kwargs.get('potential',False):
                ax2.plot(X,self.potential(X,t),linestyle='dashed')
            t0 = t
            kwargs['P0'] = P

        ax.grid()
        ax.set_xlabel('$x$')
        ax.set_ylabel(kwargs.get('ylabel','$P(x,t)$'))
        plt.title('$\epsilon='+str(self.D0)+'$')
        ax.legend(**(kwargs.get('legend_args',{})))
        plt.show()

    # First passage time problems:
    def firstpassagetime(self,x0,t0,A,**kwargs):
        """ Computes the first passage time, defined by $\tau_A = inf{t>t0 | x(t)>A}$, for one realization """
        x = x0
        t = t0
        dt = kwargs.get('dt',self.default_dt)
        while (x <= A):
            x += self.increment(x,t,dt=dt)
            t += dt
        return t

    def escapetime_sample(self,x0,t0,A,**kwargs):
        """ Computes realizations of the first passage time, defined by $\tau_A = inf{t>t0 | x(t)>A}$, using direct Monte-Carlo simulations.
        This method can be overwritten by subclasses to call compiled code for better performance.
        """
        ntraj   = kwargs.pop('ntraj',100000)
        dtype   = kwargs.pop('dtype',np.float32)
        return np.array([firstpassagetime(x0,t0,A,**kwargs) for n in xrange(ntraj)],dtype=dtype)

    def escapetime_avg(self,x0,t0,A,**kwargs):
        """ Compute the average escape time for given initial condition (x0,t0) and threshold A """
        return np.mean(self.escapetime_sample(x0,t0,A,**kwargs))

    @classmethod
    def escapetime_pdf(self,samples,**kwargs):
        """ Compute the probability distribution function of the first-passage time based on the input samples """
        avg,std = {True: (np.mean(samples),np.std(samples)), False: (0.,1.0)}.get(kwargs.get('standardize',False))
        hist, rc = np.histogram((samples-avg)/std,bins=kwargs.get('bins','doane'),density=True)
        rc = rc[:-1] + 0.5*(rc[1]-rc[0])
        return rc, hist

    @classmethod
    def escapetime_pdfplot(self,*args,**kwargs):
        """ Plot previously computed pdf of first passage time """
        fig = plt.figure()
        ax = plt.axes()
        lines = []
        for t,p in args:
            lines += ax.plot(t,p,linewidth=2)

        ax.grid()
        ax.set_xlabel(r'$\tau_M$')
        ax.set_ylabel(r'$p(\tau_M)$')
        ax.set_yscale(kwargs.get('yscale','linear'))

        plottitle = kwargs.get('title',"")
        if plottitle != "":
            plt.title(plottitle)

        labels = kwargs.get('labels',[])
        if labels != []:
            plt.legend(lines,labels,bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)

        plt.show()


    def firstpassagetime_cdf(self,x0,A,*args,**kwargs):
        """ Computes the CDF of the first passage time, Prob_{x0,t0}[\tau_A<t] by solving the Fokker-Planck equation """
        t0 = kwargs.pop('t0',0.0)
        if 'P0' in kwargs: del kwargs['P0']
        if 'P0center' in kwargs: del kwargs['P0center']
        if 'bc' not in kwargs: kwargs['bc'] = ('reflecting','absorbing')
        bnds = (kwargs.pop('bounds',(-10.0,0.0))[0],A)
        time = np.sort([t0]+list(args))
        time = time[time >= t0]
        G = [1.0 if x0 < A else 0.0]
        t,X,P = self.fpintegrate(t0,0.0,P0='dirac',P0center=x0,bounds=bnds,**kwargs)
        for t in time[1:]:
            t,X,P = self.fpintegrate(t0,t-t0,P0=P,bounds=bnds,**kwargs)
            G += [integrate.trapz(P[X<A],X[X<A])]
            t0 = t
        G = np.array(G)
        output = {'cdf': (time,1.0-G), 'G': (time,G), 'pdf': (time[1:-1],-edpy.CenteredFD(time).grad(G)), 'lambda': (time[1:-1],-edpy.CenteredFD(time).grad(np.log(G)))}
        return output.get(kwargs.get('out','G'))

    def firstpassagetime_moments(self,x0,A,*args,**kwargs):
        """ Computes the moments of the first passage time, $\langle \tau_A^n \rangle_{x0,t0}$, by solving the Fokker-Planck equation """
        t0   = kwargs.get('t0',0.0)
        tmax = kwargs.pop('tmax',10.0)
        nt   = kwargs.pop('nt',10)
        times = np.linspace(t0,tmax,num=nt)
        t,cdf = self.firstpassagetime_cdf(x0,A,*times,out='cdf',**kwargs)
        Mn  = []
        for n in args:
            Mn += [t0**n + n*integrate.trapz(cdf*times**(n-1),times)]
        return Mn

    def firstpassagetime_avg(self,x0,*args,**kwargs):
        """
        Compute the mean first passage time by one of the following methods: solving the FP equation, its adjoint, or using the theoretical solution.
        x0 is the initial condition (at t0), and 'args' contains the list of threshold values for which to compute the first passage time.
        The theoretical formula is valid only for an homogeneous process; for the computation, we 'freeze' the potential at t=t0.
        """
        src  = kwargs.pop('src','FP')
        tmax = kwargs.pop('tmax',100.0)
        nt   = kwargs.pop('nt',10)
        t0   = kwargs.pop('t0',0.0)
        inf  = kwargs.pop('inf',-10.0)
        # args have to be sorted in increasing order:
        args = np.sort(args)
        # remove the values of args which are <= x0:
        badargs,args = (args[args<=x0],args[args>x0])
        if src == 'theory':
            def exppot_int(a,b,sign=-1,fun=lambda z: 1):
                z = np.linspace(a,b)
                return integrate.trapz(np.exp(sign*self.potential(z,t0)/self.D0)*fun(z),z)
            # compute the inner integral and interpolate:
            y = np.linspace(x0,args[-1])
            arr = np.array([exppot_int(*u) for u in [(inf,y[0])]+zip(y[:-1],y[1:])])
            ifun = interp1d(y,arr.cumsum())
            # now compute the outer integral by chunks
            return np.concatenate((badargs,args)),np.array(len(badargs)*[0.]+[exppot_int(*bds,sign=1,fun=ifun) for bds in [(x0,args[0])]+zip(args[:-1],args[1:])]).cumsum()/self.D0
        elif src == 'theory2':
            def exppot(y,sign=-1,fun=lambda z: 1):
                return np.exp(sign*self.potential(y,t0)/self.D0)*fun(y)
            # compute the inner integral and interpolate:
            z = np.linspace(inf,args[-1])
            iarr = integrate.cumtrapz(exppot(z),z,initial=0)
            ifun = interp1d(z,iarr)
            # now compute the outer integral by chunks
            y = np.linspace(x0,args[-1])
            oarr = integrate.cumtrapz(exppot(y,sign=1,fun=ifun),y,initial=0)/self.D0
            ofun = interp1d(y,oarr)
            return np.concatenate((badargs,args)),np.concatenate((len(badargs)*[0.],ofun(args)))
        elif src == 'adjoint':
            # here we need to solve the adjoint FP equation for each threshold value, so this is much more expensive than the theoretical formula of course.
            def interp_int(G,t):
                logG = interp1d(t,np.log(G),fill_value="extrapolate")
                return integrate.quad(lambda x: np.exp(logG(x)),0.0,np.inf)[0] # careful: this is not the right expression for arbitrary t0 !!
            integ_method = {True: interp_int, False: integrate.trapz}.get(kwargs.pop('interpolate',True))
            return np.concatenate((badargs,args)),np.array(len(badargs)*[0.]+[integ_method(*(self.firstpassagetime_cdf(x0,A,*np.linspace(0.0,tmax,num=nt),t0=t0,out='G',src='adjoint',**kwargs)[::-1])) for A in args])
#        elif src in ('FP','quad'):
        elif src == 'FP':
            # here we need to solve the FP equation for each threshold value, so this is much more expensive than the theoretical formula of course.
            def interp_int(G,t):
                logG = interp1d(t,np.log(G),fill_value="extrapolate")
                return integrate.quad(lambda x: np.exp(logG(x)),t0,np.inf)[0]
            integ_method = {True: interp_int, False: integrate.trapz}.get(kwargs.pop('interpolate',True))
            return np.concatenate((badargs,args)),np.array(len(badargs)*[0.]+[t0+integ_method(*(self.firstpassagetime_cdf(x0,A,*np.linspace(t0,tmax,num=nt),t0=t0,out='G',**kwargs)[::-1])) for A in args])
        else:
            pass

class Wiener(StochModel):
    """ The Wiener process """
    def __init__(self,D=1):
        super(self.__class__,self).__init__(lambda x,t: 0, D)

    def potential(self,X,t):
        """ Useless (and potentially source of errors) to call the general potential routine since it is trivially zero here """
        return np.zeros_like(X)

    def _fpthsol(self,X,t,**kwargs):
        """ Analytic solution of the heat equation.
        This should depend on the boundary conditions.
        Right now, we do as if we were solving on the real axis."""
        return np.exp(-X**2.0/(4.0*self.D0*t))/np.sqrt(4.0*np.pi*self.D0*t)


class OrnsteinUhlenbeck(StochModel):
    """ The Ornstein-Uhlenbeck model """
    def __init__(self,mu,theta,D):
        super(self.__class__,self).__init__(lambda x,t: theta*(mu-x),D)


class DrivenDoubleWell(StochModel):
    """ Double well potential model, possibly including periodic forcing and noise """

    default_dt = 0.01

    def __init__(self,Famp,omega,Damp):
        super(DrivenDoubleWell,self).__init__(lambda x,t: -x*(x**2-1)+Famp*np.sin(omega*t),Damp)
        self.Famp = Famp
        self.Om   = omega

    def potential(self,X,t):
        """ Return the value of the potential at the input points """
        Y = X**2
        return Y*(Y-2.0)/4.0-X*self.Famp*np.sin(self.Om*t)

    def phaseportrait(self,a,b,ntraj,niter,dt):
        """ Compute and plot the trajectories for an ensemble of initial conditions """
        for x0 in np.linspace(a,b,num=ntraj):
            plt.plot(*self.trajectory(x0,0,T=niter*dt,dt=dt))
        plt.xlabel('$t$')
        plt.ylabel('$x(t)$')
        plt.show()

    @classmethod
    def _trajectoryplot_decorate(cls,*args,**kwargs):
        """ Plot the fixed point trajectories """
        ax = kwargs.get('axis')
        ax.axhline(y=1.0,linewidth=1,color='black')
        ax.axhline(y=0.0,linewidth=1,color='black',linestyle='dashed')
        ax.axhline(y=-1.0,linewidth=1,color='black')
        # tmin = kwargs.get('tmin',np.min(zip(*args)[0]))
        # tmax = kwargs.get('tmax',np.max(zip(*args)[0]))
        # ax.set_xlim([tmin,tmax])


    def running_mean(x, N):
        cumsum = np.cumsum(np.insert(x, 0, 0))
        return (cumsum[N:] - cumsum[:-N]) / N

    def transitionrate(self,x,**kwargs):
        """ Count the number of transitions from one attractor to the other for a given trajectory.
            Without smoothing (avg=1), that should be the number of items in the generator levelscrossing(x,0) when starting with the right transition,
            or that number +1 if we use the wrong isign in levelscrossing."""
        window = kwargs.get('avg',1)
        y = self.running_mean(x,window) if window > 1 else x
        return float((y[1:]*y[:-1] < 0).sum())/len(y)

    def levelscrossing(self,x,c,**kwargs):
        """ Maps the stochastic process x(t) onto a stochastic process {t_i} where the 't_i's correspond to crossing levels +- c """
        sign = kwargs.get('isign',1) # By default we start by detecting the transition below the -c threshold
        if sign == 0: sign=1
        if not abs(sign) == 1: sign /= abs(sign)
        for i in xrange(len(x)-1):
            if (c+sign*x[i]) > 0 and (c+sign*x[i+1]) < 0:
                sign *= -1
                yield i

    def residencetimes(self,x,c):
        transtimes = np.array([t for t in self.levelscrossing(x,c)])
        return transtimes[1:]-transtimes[:-1]


    def escapetime_sample(self,x0,t0,A,**kwargs):
        """ This is a wrapper to the compiled vec_escape_time_dw function.
        It carries out direct Monte-Carlo simulations to compute realizations of the stochastic process.
        """
        # Input parameters and options:
        dt      = kwargs.get('dt',self.default_dt)
        ntraj   = kwargs.get('ntraj',100000)
        dtype   = kwargs.get('dtype',np.float32)
        return vec_escape_time_dw(
                np.full(ntraj,x0,dtype=dtype),
                np.full(ntraj,t0,dtype=dtype),
                np.full(ntraj,A,dtype=dtype),
                np.full(ntraj,dt,dtype=dtype),
                np.full(ntraj,self.D0,dtype=dtype))


    def fpintegrate(self,t0,T,**kwargs):
        """ Numerical integration of the associated Fokker-Planck equation.
        We only modify the default values for a few parameters, to adjust to the specificities of the model:
        - Default boundaries do not need to be so wide (potential is very steep)
        - Default initial condition is (a Gaussian) centered on one of the wells (-1) with smaller standard deviation
        - Default boundary conditions are reflecting on both sides """
        return super(DrivenDoubleWell,self).fpintegrate(t0,T,bounds=kwargs.pop('bounds',(-3.0,3.0)),P0center=kwargs.pop('P0center',-1.0),P0std=kwargs.pop('P0std',0.1),bc=kwargs.pop('bc',('reflecting','reflecting')),**kwargs)

    def fpadjintegrate(self,t0,T,**kwargs):
        """ Numerical integration of the adjoint Fokker-Planck equation.
        Be careful that we are relying on the fpintegrate method from the superclass,
        which was originally designed to integrate the forward FP equation.
        But since the routines are so similar, it does not make sense to duplicate the code.
        Just keep it in mind when modifying the base class. """
        # Set initial G(x)
        # I still need to define the grid at this level to build the IC...
        B,A    = kwargs.pop('bounds',(-3.0,0.0))
        fdgrid = edpy.RegularCenteredFD(B,A,kwargs.get('npts',100))
        G0     = np.ones_like(fdgrid.grid)
        G0[fdgrid.grid >= A] = 0.0
        # Call integration routine with adjoint FP equation:
        return super(DrivenDoubleWell,self).fpintegrate(t0,T,bounds=(B,A),P0=kwargs.pop('P0',G0),bc=kwargs.pop('bc',edpy.BoundaryCondition(lambda Y,X,t: [Y[1],0])),adjoint=True,**kwargs)

    def firstpassagetime_cdf(self,x0,A,*args,**kwargs):
        """ Computes the CDF of the first passage time, Prob_{x0,t0}[\tau_A<t], either by solving the Fokker-Planck equation, its adjoint, or by using the theoretical solution. """
        src = kwargs.pop('src','FP')
        if src == 'theory':
            # Poissonian statistics for the first passage time:
            t = np.array(args)
            Lambda = 1./(self.firstpassagetime_avg(x0,A,src='theory')[1])
            G = np.exp(-Lambda*t)
            P = Lambda*G
            # t = t[t<0]
            return t,{'cdf': 1.0-G, 'G': G, 'pdf': P, 'lambda': Lambda}.get(kwargs.get('out','G'))
        elif src == 'adjoint':
            t0 = kwargs.pop('t0',0.0)
            # The upper boundary of the integration domain should always be A
            # Hence, here only the lower end of the user-provided domain is used (we still need the two things for the FP method)
            bnds = (kwargs.pop('bounds',(-3.0,0.0))[0],A)
            time = np.sort([t0]+list(args))
            time = time[time >= t0]
            G = [1.0]
            for t in time[1:]:
                t,X,G1 = self.fpadjintegrate(t0,t-t0,bounds=bnds,**kwargs)
                t0 = t
                kwargs['P0'] = G1
                G += [interp1d(X,G1)(x0)]
            G = np.array(G)
            output = {'cdf': (time,1.0-G), 'G': (time,G), 'pdf': (time[1:-1],-edpy.CenteredFD(time).grad(G)), 'lambda': (time[1:-1],-edpy.CenteredFD(time).grad(np.log(G)))}
            return output.get(kwargs.get('out','G'))
        else:
            return super(DrivenDoubleWell,self).firstpassagetime_cdf(x0,A,*args,bc=kwargs.pop('bc',('reflecting','absorbing')),**kwargs)

    def firstpassagetime_avg(self,x0,*args,**kwargs):
        """
        Compute the mean first passage time by one of the following methods: solving the FP equation, its adjoint, or using the theoretical solution.
        x0 is the initial condition (at t0=0), and 'args' contains the list of threshold values for which to compute the first passage time.
        """
        src  = kwargs.pop('src','FP')
        tmax = kwargs.pop('tmax',100.0)
        nt   = kwargs.pop('nt',10)
        # args have to be sorted in increasing order:
        args = np.sort(args)
        # remove the values of args which are <= x0:
        badargs,args = (args[args<=x0],args[args>x0])
        if src == 'theory':
            def exppot_int(a,b,sign=-1,fun=lambda z: 1):
                z = np.linspace(a,b)
                return integrate.trapz(np.exp(sign*self.potential(z,0)/self.D0)*fun(z),z)
            # compute the inner integral and interpolate:
            y = np.linspace(x0,args[-1])
            arr = np.array([exppot_int(*u) for u in [(-10.0,y[0])]+zip(y[:-1],y[1:])])
            ifun = interp1d(y,arr.cumsum())
            # now compute the outer integral by chunks
            return np.concatenate((badargs,args)),np.array(len(badargs)*[0.]+[exppot_int(*bds,sign=1,fun=ifun) for bds in [(x0,args[0])]+zip(args[:-1],args[1:])]).cumsum()/self.D0
        elif src == 'theory2':
            def exppot(y,sign=-1,fun=lambda z: 1):
                return np.exp(sign*self.potential(y,0)/self.D0)*fun(y)
            # compute the inner integral and interpolate:
            z = np.linspace(-10.0,args[-1])
            iarr = integrate.cumtrapz(exppot(z),z,initial=0)
            ifun = interp1d(z,iarr)
            # now compute the outer integral by chunks
            y = np.linspace(x0,args[-1])
            oarr = integrate.cumtrapz(exppot(y,sign=1,fun=ifun),y,initial=0)/self.D0
            ofun = interp1d(y,oarr)
            return np.concatenate((badargs,args)),np.concatenate((len(badargs)*[0.],ofun(args)))
        elif src == 'adjoint':
            # here we need to solve the adjoint FP equation for each threshold value, so this is much more expensive than the theoretical formula of course.
            def interp_int(G,t):
                logG = interp1d(t,np.log(G),fill_value="extrapolate")
                return integrate.quad(lambda x: np.exp(logG(x)),0.0,np.inf)[0]
            integ_method = {True: interp_int, False: integrate.trapz}.get(kwargs.pop('interpolate',True))
            return np.concatenate((badargs,args)),np.array(len(badargs)*[0.]+[integ_method(*(self.firstpassagetime_cdf(x0,A,*np.linspace(0.0,tmax,num=nt),t0=0.0,out='G',src='adjoint',**kwargs)[::-1])) for A in args])
        else:
            pass

    def instanton(self,x0,p0,*args,**kwargs):
        def fun(Y,t):
            return (Y[0]*(1.-Y[0]**2)+2.*Y[1]+self.Famp*np.sin(self.Om*t),Y[1]*(3.*Y[0]**2-1.))
        return integrate.odeint(fun,(x0,p0),args,**kwargs)

class DoubleWell(DrivenDoubleWell):
    """ The classical double-well potential model with noise """
    def __init__(self,Damp):
        super(DoubleWell,self).__init__(0.0,0.0,Damp)

class StochSaddleNode(StochModel):
    """ This is a very simple model for loss of stability with noise.
    We use the normal form of the saddle-node bifurcation with a time dependent parameter, and add a stochastic term. """

    default_dt = 0.01

    def __init__(self,Damp):
        super(StochSaddleNode,self).__init__(lambda x,t: x**2+t,Damp)

    def potential(self,X,t):
        """ Return the value of the potential at the input points """
        return -X**3/3.0-t*X

    @classmethod
    def _trajectoryplot_decorate(cls,*args,**kwargs):
        """ Plot the fixed point trajectories """
        ax = kwargs.get('axis')
        ax.set_ylim(kwargs.get('ylim',(-10.0,10.0)))
        tmin = min([min(t) for t,x in args])
        time = np.linspace(tmin,0,num=max(50,5*np.floor(abs(tmin))))
        plt.plot(time,-np.sqrt(np.abs(time)),color='black')
        plt.plot(time,np.sqrt(np.abs(time)),linestyle='dashed',color='black')

    def trajectory(self,x0,t0,**kwargs):
        """ This is a wrapper to the compiled saddlenode_trajectory function """
        time = kwargs.get('T',10)   # Total integration time
        dt   = kwargs.get('dt',self.default_dt) # Time step
        if dt == 'adapt':
            t,x = saddlenode_trajectory_adapt(x0,self.D0,t0,time,self.default_dt)
        else:
            if dt == 'auto': dt = min(self.default_dt,0.1/np.sqrt(np.abs(t0)))
            if dt < 0: time = -time
            t = np.linspace(t0,t0+time,num=time/dt+1)
            x = saddlenode_trajectory(x0,dt,self.D0,t[1:])
        if kwargs.get('finite',False):
            t = t[np.isfinite(x)]
            x = x[np.isfinite(x)]
        return t,x

    def escapetime_sample(self,x0,t0,A,**kwargs):
        """ This is a wrapper to the compiled vec_escape_time function """
        # Input parameters and options:
        dt      = kwargs.get('dt',self.default_dt)
        ntraj   = kwargs.get('ntraj',100000)
        dtype   = kwargs.get('dtype',np.float32)
        if dt == 'adapt':
            fun = vec_escape_time_adapt
            dt  = self.default_dt
        else:
            if dt == 'auto': dt = min(self.default_dt,0.1/np.sqrt(np.abs(t0)))
            fun = vec_escape_time
        # If we are using the caching mechanism, we check how many realizations we need to compute for the set of input parameters:
        ncache = 0
        if kwargs.get('cache',False):
            db = data.Database("/Users/corentin/data/stochtrans/tau.db")
            sample_cache = db[self.D0,t0,x0,kwargs.get('dt',self.default_dt),A]
            ncache = len(sample_cache)
        # We compute the remaining realizations:
        if ntraj>ncache:
            sample = fun(
                np.full(ntraj-ncache,x0,dtype=dtype),
                np.full(ntraj-ncache,t0,dtype=dtype),
                np.full(ntraj-ncache,A,dtype=dtype),
                np.full(ntraj-ncache,dt,dtype=dtype),
                np.full(ntraj-ncache,self.D0,dtype=dtype))
            # Store the new realizations to cache
            if kwargs.get('cache',False):
                sample_cache = db[self.D0,t0,x0,kwargs.get('dt',self.default_dt),A] = np.concatenate((sample_cache,sample))
            else:
                sample_cache = sample
        return sample_cache[:ntraj]


    def fpintegrate(self,t0,T,**kwargs):
        """ Numerical integration of the associated Fokker-Planck equation.
        We only modify the default values for a few parameters, to adjust to the specificities of the model:
        - Default initial condition is (a Gaussian) centered on the attractor
        - Default boundary conditions are reflecting on the left and absorbing on the right (potential does not go up on that side)
        Default boundaries, etc, are fine """
        return super(self.__class__,self).fpintegrate(t0,T,P0center=kwargs.pop('P0center',-np.sqrt(np.abs(t0))),bc=kwargs.pop('bc',('reflecting','absorbing')),**kwargs)

    def pdfplot(self,*args,**kwargs):
        """ Plot the pdf P(x,t) at various times """
        t0 = kwargs.get('t0',args[0])
        super(self.__class__,self).pdfplot(*args,P0=kwargs.pop('P0','gauss'),P0center=kwargs.pop('P0center',-np.sqrt(np.abs(t0))),**kwargs)


    def firstpassagetime_cdf(self,x0,A,*args,**kwargs):
        """ Computes the CDF of the first passage time, Prob_{x0,t0}[\tau_A<t], either by solving the Fokker-Planck equation, or by using the Eyring-Kramers ansatz. """
        if kwargs.get('EK',False)==True:
            t = np.array(args)
            t = t[t<0]
            # this seems to be wrong: I guess I used V(a,b) instead of V''(a,b) in the prefactor
            #G = np.exp(-(3./4.)**(2./3.)*(self.D0)**(5./3.)*gammaincc(5./3.,4.*(-t)**1.5/(3.*self.D0))*gamma(5./3.)/(6.*np.pi))
            #Lambda = (-t)**(1.5)*np.exp(-4.*(-t)**1.5/(3.*self.D0))/(3*np.pi)
            G = np.exp(-self.D0/(2*np.pi)*np.exp(-4.*(-t)**1.5/(3.*self.D0)))
            Lambda = np.sqrt(-t)*np.exp(-4.*(-t)**1.5/(3.*self.D0))/np.pi
            P = Lambda*G/(1.0-np.exp(-self.D0/(2*np.pi)))
            avg, std = (0.0,1.0)
            if kwargs.get('standardize',False):
                avg,std = (integrate.trapz(t*P,t),integrate.trapz(t**2*P,t))
                std = np.sqrt(std-avg**2)
            return {'cdf': (t,1.0-G), 'G': (t,G), 'pdf': ((t-avg)/std,std*P), 'lambda': (t,Lambda)}.get(kwargs.get('out','G'))
        else:
            if kwargs.get('EK',False)=='quad':
                t = np.array(args)
                fixedx0 = kwargs.pop('fixedx0',False)
                Lambda = np.array([1./self.firstpassagetime_avg((x0 if fixedx0 else -np.sqrt(np.abs(t0))),A,src='theory',t0=t0,**kwargs)[1][0] for t0 in t])
                G = np.exp(-integrate.cumtrapz(Lambda,t,initial=0))
                P = Lambda*G
                avg = 0.0
                std = 1.0
                if kwargs.get('standardize',False):
                    avg = integrate.trapz(t*P,t)
                    std = integrate.trapz(t**2*P,t)
                    std = np.sqrt(std-avg**2)
                return {'cdf': (t,1.0-G), 'G': (t,G), 'pdf': ((t-avg)/std,std*P), 'lambda': (t,Lambda)}.get(kwargs.get('out','G'))
            else:
                return super(self.__class__,self).firstpassagetime_cdf(x0,A,*args,**kwargs)

    def firstpassagetime_avg(self,x0,*args,**kwargs):
        """
        Compute mean first-passage time with Eyring-Kramers ansatz or using the superclass general techniques.
        Caveat: the Eyring-Kramers formula does not depend on the threshold value M ! The function prototype with variable number of arguments is kept for compatibility purposes, but it only makes sense to call it with one argument M.
        """
        if kwargs.get('src','FP')=='EK':
            def efun(u,n):
                return u**(2.*n/3.)*np.exp(-u)*np.exp(-self.D0/(2*np.pi)*np.exp(-u))
            # args have to be sorted in increasing order:
            args = np.sort(args)
            # remove the values of args which are <= x0:
            badargs,args = (args[args<=x0],args[args>x0])
            t0   = kwargs.pop('t0',0.0)
            n    = kwargs.pop('order',1)
            k = (0.75*self.D0)**(2./3.)
            tau = (-1)**n*self.D0/(2.*np.pi)*k**n*integrate.quad(efun,0.,min((-t0/k)**1.5,100.0),args=(n,))[0]/(1.0-np.exp(-self.D0/(2.*np.pi)))
            return np.concatenate((badargs,args)),np.array(len(badargs)*[0.]+len(args)*[tau])
        #elif kwargs.get('src','FP')=='quad':
        #    return super(self.__class__,self).firstpassagetime_avg(x0,*args,EK='quad',**kwargs)
        else:
            return super(self.__class__,self).firstpassagetime_avg(x0,*args,**kwargs)

    @classmethod
    def instanton(cls,x0,p0,*args,**kwargs):
        """
        Numerical integration of the equations of motion for instantons.
        x0 and p0 are the initial conditions.
        Return the instanton trajectory (t,x).
        """
        def fun(Y,t):
            """Equations of motion for instanton dynamics."""
            return [Y[0]**2+2.*Y[1]+t,-2.*Y[0]*Y[1]]
        def jac(Y,t):
            """Jacobian of instanton dynamics."""
            return [[2*Y[0],2.0],[-2.0*Y[1],-2.0*Y[0]]]
        def fun_log(Y,t):
            """Vector field with the change of variable q=ln(p)."""
            return [Y[0]**2+2.*np.exp(Y[1])+t,-2.*Y[0]]
        def jac_log(Y,t):
            """Jacobian of the vector field with the change of variable q=ln(p)."""
            return [[2.*Y[0],2.*np.exp(Y[1])],[-2.0,0.0]]
        def rev(f):
            return lambda t,Y: f(Y,t)
        def filt_fun(t,x):
            filt = (x>100.0).nonzero()[0]
            if len(filt) >0:
                maxind = filt[0]
            else:
                maxind = -1
            return t[:maxind], x[:maxind]
        solver    = kwargs.pop('solver','odeint')
        scheme    = kwargs.pop('integrator','dopri5')
        var       = kwargs.pop('change_vars','None')
        filt_traj = kwargs.pop('filter_traj',False)
        times = np.sort(args)
        if var == 'log':
            fun = fun_log
            jac = jac_log
        if solver == 'odeint':
            x = integrate.odeint(fun,(x0,p0),times,**kwargs)[:,0]
            return filt_fun(times,x) if filt_traj else (times,x)
        elif solver == 'odeclass':
            integ = integrate.ode(rev(fun),jac=rev(jac)).set_integrator(scheme,**kwargs)
            integ.set_initial_value([x0,p0],times[0])
            return times,[integ.integrate(t)[0] for t in times]

    @classmethod
    def instanton_cond(cls,x0,t0,T,M,**kwargs):
        """
        Compute the instanton given by the initial condition (x0,t0), the duration T and the final position M.
        """
        def duration_solve(q0,t0,T,M):
            times = np.arange(t0,t0+1.5*T,0.05)
            t,x = cls.instanton(-np.sqrt(np.abs(t0)),q0,*times,solver='odeclass',change_vars='log')
            return t[x>=M][0]-t0-T
        def position_solve(q0,t0,T,M):
            t,x = cls.instanton(-np.sqrt(np.abs(t0)),q0,t0,t0+T,solver='odeclass',change_vars='log')
            return x[-1]-M
        fun   = {'duration': duration_solve, 'position': position_solve}.get(kwargs.pop('method','position'))
        qmin  = kwargs.pop('qmin',-1000.0)
        qmax  = kwargs.pop('qmax',-1.0)
        qsol  = brentq(fun,qmin,qmax,args=(t0,T,M))
        times = kwargs.pop('times',np.linspace(t0,t0+T,num=kwargs.pop('npts',100)))
        return cls.instanton(x0,qsol,*times,solver='odeclass',change_vars='log',**kwargs)

class DynSaddleNode(StochSaddleNode):
    """ This is just the deterministic version of the dynamical saddle-node bifurcation dx/dt = x^2+t, for which we have an analytic solution """

    tstar = -ai_zeros(1)[0][0]

    def __init__(self):
        super(self.__class__,self).__init__(0)

    @classmethod
    def formula(cls,t,A0,B0):
        ai,aip,bi,bip = airy(-t)
        return (A0*aip+B0*bip)/(A0*ai+B0*bi)

    @classmethod
    def coefficients(cls,x0,t0):
        ai,aip,bi,bip = airy(-t0)
        alpha, beta = (x0*ai-aip,x0*bi-bip)
        if alpha == 0:
            A0, B0 = (1,0)
        else:
            A0, B0 = (-beta/alpha,1)
        return A0,B0

    @classmethod
    def trajectory(cls,x0,t0,**kwargs):
        """ Analytic solution for trajectories """
        dt     = kwargs.get('dt',cls.default_dt) # Time step
        time   = kwargs.get('T',10)   # Total integration time
        if dt < 0: time=-time
        tarray = np.linspace(t0,t0+time,num=time/dt+1)
        return tarray,cls.formula(tarray,*cls.coefficients(x0,t0))

    @classmethod
    def firstpassagetime(cls,x0,t0,A,**kwargs):
        """ Compute first passage time using the analytic solution """
        return brentq(lambda t: cls.formula(t,*cls.coefficients(x0,t0))-A,t0,cls.tstar)


###
#
#  Compiled code using numba for better performance
#
###

@vectorize(['float32(float32,float32,float32,float32,float32)','float64(float64,float64,float64,float64,float64)'],target='parallel')
def vec_escape_time(x0,t0,A,dt,D0):
    """ Computes the escape time, defined by inf{t>t0 | x(t)>A}, for one realization """
    x = x0
    t = t0
    while (x <= A):
        x += (x**2+t) * dt + np.sqrt(2*D0*dt)*np.random.normal(0.0,1.0)
        t += dt
    return t

@vectorize(['float32(float32,float32,float32,float32,float32)','float64(float64,float64,float64,float64,float64)'],target='parallel')
def vec_escape_time_adapt(x0,t0,A,dtmax,D0):
    """ Computes the escape time, defined by inf{t>t0 | x(t)>A}, for one realization, with adaptive timestep """
    x = x0
    t = t0
    while (x <= A):
        dt = min(dtmax,0.1/np.sqrt(np.abs(t)))
        x += (x**2+t) * dt + np.sqrt(2*D0*dt)*np.random.normal(0.0,1.0)
        t += dt
    return t

@vectorize(['float32(float32,float32,float32,float32,float32)','float64(float64,float64,float64,float64,float64)'],target='parallel')
def vec_escape_time_dw(x0,t0,A,dt,D0):
    """ Computes the escape time, defined by inf{t>t0 | x(t)>A}, for one realization, for the double well potential """
    x = x0
    t = t0
    while (x <= A):
        x += - x * (x**2-1) * dt + np.sqrt(2*D0*dt)*np.random.normal(0.0,1.0)
        t += dt
    return t


@jit(["float32[:](float32,float32,float32,float32[:])","float64[:](float64,float64,float64,float64[:])"],target='cpu',nopython=True)
def saddlenode_trajectory(x0,dt,D0,tarr):
    """ Integrate a trajectory with given initial condition (t0,x0) """
    x = [x0]
    for t in tarr:
        x += [ x[-1] + (x[-1]**2+t) * dt + np.sqrt(2*D0*dt)*np.random.normal(0.0,1.0)]
    return np.array(x)


@jit(["float32[:,:](float32,float32,float32,float32,float32)","float64[:,:](float64,float64,float64,float64,float64)"],target='cpu',nopython=True)
def saddlenode_trajectory_adapt(x0,D0,t0,T,dtmax):
    """ Integrate a trajectory with given initial condition (t0,x0) and adaptive timestep """
    x = [x0]
    t = [t0]
    while t[-1] < t0+T:
        dt = min(dtmax,0.1/np.sqrt(np.abs(t[-1])))
        x += [ x[-1] + (x[-1]**2+t[-1]) * dt + np.sqrt(2*D0*dt)*np.random.normal(0.0,1.0)]
        t += [ t[-1] + dt ]
    return np.array((t,x))



class StochModel_T(StochModel):
    """ Time reversal of a given model """
    def trajectory(self,x0,t0,**kwargs):
        t,x = super(self.__class__,self).trajectory(x0,t0,**kwargs)
        return 2*t[0]-t,x
