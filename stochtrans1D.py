import numpy as np
import matplotlib.pyplot as plt
from numba import float32,float64,vectorize,autojit,jit

class StochModel(object):
    """ The generic class from which all the models I consider derive.
        It corresponds to the family of 1D SDEs dx_t = F(x_t,t)dt + sqrt(2*D0)dW_t """
    def __init__(self,vecfield,Damp):
        """ vecfield is a function of two variables (x,t) and Damp the amplitude of the diffusion term (noise) """
        self.F  = vecfield 
        self.D0 = Damp

    def time_reversal(self):
        """ Apply time reversal and return the new model """
        return StochModel_T(lambda x,t: -self.F(x,-t),self.D0)
        
    def trajectory(self,x0,t0,**kwargs):
        """ Integrate a trajectory with given initial condition (t0,x0) """
        x      = [x0]
        dt     = kwargs.get('dt',0.1) # Time step
        time   = kwargs.get('T',10)   # Total integration time
        if dt < 0: time=-time
        tarray = np.linspace(t0,t0+time,num=time/dt+1)
        for t in tarray[1:]:    
            x += [ x[-1] + self.F(x[-1],t) * dt + np.sqrt(2*self.D0*dt)*np.random.normal(0.0,1.0)]
        x = np.array(x)
        if kwargs.get('finite',False):            
            tarray = tarray[np.isfinite(x)]
            x = x[np.isfinite(x)]
        return tarray,x

    @classmethod
    def trajectoryplot(cls,*args,**kwargs):        
        """ Plot previously computed trajectories """
        fig = plt.figure()
        ax  = plt.axes()
        lines = []
        for t,x in args:
            lines += ax.plot(t,x)        
        
        ax.grid()
        ax.set_ylim(kwargs.get('ylim',(-10.0,10.0)))
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

    def fpintegrate(self,t0,T,B,A,Np,dt,**kwargs):
        """ Numerical integration of the associated Fokker-Planck equation """
        
        fdgrid = RegularCenteredFD(B,A,Np)    
    
        # initial P(x)    
        P = kwargs.get('P',np.exp(-0.5*(fdgrid.grid+np.sqrt(np.abs(t0)))**2)/np.sqrt(2*np.pi))    

        # time integration
        t = t0
        while (t < t0+T):
            # Advancing in the bulk:
            deltaP = fdgrid.laplacian(P)
            drift = fdgrid.grad(self.F(fdgrid.grid,t)*P)
            P[1:-1] += (-drift + self.D0*deltaP)*dt

            # Absorbing BC at x=A:
            P[-1] = 0
            # # Reflecting BC at x=B:
            # P[0] = P[1]/(1+self.F(fdgrid.grid[0],t)*fdgrid.dx/self.D0)
            P[0] = 0
    
            t += dt

        return t,P
    
class Wiener(StochModel):
    """ The Wiener process """
    def __init__(self):
        super(self.__class__,self).__init__(lambda x,t: 0, 1)


class OrnsteinUhlenbeck(StochModel):
    """ The Ornstein-Uhlenbeck model """
    def __init__(self,mu,theta,D):
        super(self.__class__,self).__init__(lambda x,t: theta*(mu-x),D)
        
    
class DoubleWell(StochModel):
    """ Double well potential model, possibly including periodic forcing and noise """

    def __init__(self,Famp,omega,Damp):
        super(self.__class__,self).__init__(lambda x,t: -x*(x**2-1)+Famp*np.sin(omega*t),Damp)

    def phaseportrait(self,a,b,ntraj,niter,dt):
        """ Compute and plot the trajectories for an ensemble of initial conditions """
        time = np.linspace(0,(niter-1)*dt,num=niter)
        for x0 in np.linspace(a,b,num=ntraj):
            plt.plot(time,self.trajectory(x0,0,T=niter*dt,dt=dt))        
        plt.xlabel('$t$')
        plt.ylabel('$x(t)$')
        plt.show()

    def trajectoryplot(self,*args,**kwargs):
        """ Plot previously computed trajectories with initial time t0 and time unit dt """
        dt = kwargs.get('dt',1)
        t0 = kwargs.get('t0',0)
        Tmin = kwargs.get('tmin',t0)
        tmax = t0
        for x in args:
            plt.plot(np.linspace(t0,(len(x)-1)*dt,num=len(x)),x)
            tmax = max(tmax,len(x)*dt)
        tmax = kwargs.get('tmax',tmax)
        plt.xlim([tmin,tmax])
        plt.xlabel('$t$')
        plt.ylabel('$x(t)$')
        plt.show()

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
        

class StochSaddleNode(StochModel):
    default_dt = 0.01
    
    def __init__(self,Damp):
        super(self.__class__,self).__init__(lambda x,t: x**2+t,Damp)

    @classmethod
    def _trajectoryplot_decorate(cls,*args,**kwargs):
        """ Plot the fixed point trajectories """
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


    def escapetime(self,x0,t0,A,**kwargs):
        """ Computes the escape time, defined by inf{t>t0 | x(t)>A}, for one realization """
        x = x0
        t = t0
        dt = kwargs.get('dt',self.default_dt)
        while (x <= A):
            x += self.F(x,t) * dt + np.sqrt(2*self.D0*dt)*np.random.normal(0.0,1.0)
            t += dt
        return t
    
    def escapetime_sample(self,x0,t0,A,**kwargs):
        """ This is a wrapper to the compiled vec_escape_time function """
        dt      = kwargs.get('dt',self.default_dt)
        ntraj   = kwargs.get('ntraj',100000)
        dtype   = kwargs.get('dtype',np.float32)
        if dt == 'adapt':
            fun = vec_escape_time_adapt
            dt  = self.default_dt
        else:
            if dt == 'auto': dt = min(self.default_dt,0.1/np.sqrt(np.abs(t0)))
            fun = vec_escape_time
        return fun(
            np.full(ntraj,x0,dtype=dtype),
            np.full(ntraj,t0,dtype=dtype),
            np.full(ntraj,A,dtype=dtype),
            np.full(ntraj,dt,dtype=dtype),
            np.full(ntraj,self.D0,dtype=dtype))    

    def escapetime_avg(self,x0,t0,A,**kwargs):
        """ Compute the average escape time for given initial condition (x0,t0) and threshold A """
        return np.mean(self.escapetime_sample(x0,t0,A,**kwargs))
    
    def escapetime_pdf(self,x0,t0,A,**kwargs):
        """ Compute the probability distribution function of the escape time with given initial conditions (t0,x0) and a given threshold A """
        samples = self.escapetime_sample(x0,t0,A,**kwargs)
        if (kwargs.get('standardize',False)):
            samples -= np.mean(samples)
            samples /= np.std(samples)
        hist, rc = np.histogram(samples,bins=kwargs.get('bins','doane'),density=True)
        rc = rc[:-1] + 0.5*(rc[1]-rc[0])
        return rc, hist
    
    def escapetime_pdf_plot(self,x0,t0,A,**kwargs):        
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xlabel('$t_\star$')
        ax.set_ylabel('$p(t_\star)$')
        plt.grid()
        pdf_line, = ax.plot(*self.escapetime_pdf(x0,t0,A,**kwargs),linewidth=2)
        plt.show()


    def fpintegrate(self,t0,T,B,A,Np,dt,**kwargs):
        """ Numerical integration of the associated Fokker-Planck equation """
        
        fdgrid = RegularCenteredFD(B,A,Np)    
    
        # initial P(x)    
        P = kwargs.get('P',np.exp(-0.5*(fdgrid.grid+np.sqrt(np.abs(t0)))**2)/np.sqrt(2*np.pi))    

        # time integration
        t = t0
        while (t < t0+T):
            # Advancing in the bulk:
            deltaP = fdgrid.laplacian(P)
            drift = fdgrid.grad(self.F(fdgrid.grid,t)*P)
            P[1:-1] += (-drift + self.D0*deltaP)*dt

            # Absorbing BC at x=A:
            P[-1] = 0
            # Reflecting BC at x=B:
            P[0] = P[1]/(1+self.F(fdgrid.grid[0],t)*fdgrid.dx/self.D0)
    
            t += dt

        return t,P

        
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

