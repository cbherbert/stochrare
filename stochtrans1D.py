import numpy as np
import matplotlib.pyplot as plt

class StochModel(object):
    """ The generic class from which all the models I consider derive """
    def __init__(self,vecfield,Damp):
        """ vecfield is a function of two variables (x,t) and Damp the amplitude of the diffusion term (noise) """
        self.F  = vecfield 
        self.D0 = Damp

    def trajectory(self,x0,t0,**kwargs):
        """ Integrate a trajectory with given initial condition (t0,x0), with time step dt, for niter iterations """
        x      = [x0]
        dt     = kwargs.get('dt',0.1)
        time   = kwargs.get('T',10)
        tarray = np.linspace(t0,t0+time,num=time/dt+1)
        for t in tarray[1:]:    
            x += [ x[-1] + self.F(x[-1],t) * dt + np.sqrt(2*self.D0*dt)*np.random.normal(0.0,1.0)]
        return tarray,np.array(x)


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

    def __init__(self,Damp):
        super(self.__class__,self).__init__(lambda x,t: x**2+t,Damp)

    def trajectoryplot(self,*args,**kwargs):
        """ Plot previously computed trajectories """
        for t,x in args:
            plt.plot(t,x)

        # Plot the fixed point trajectories
        tmin = min([min(t) for t,x in args])
        time = np.linspace(tmin,0)
        plt.plot(time,-np.sqrt(abs(time)),color='black')
        plt.plot(time,np.sqrt(abs(time)),linestyle='dashed',color='black')

        plt.grid()
        #plt.xlim([tmin,tmax])
        plt.ylim([-10.0,10.0])
        plt.xlabel('$t$')
        plt.ylabel('$x(t)$')
        plt.show()

    def escape_time(self,x0,t0,A,**kwargs):
        """ Computes the escape time, defined by inf{t>t0 | x(t)>A}, for one realization """
        x = x0
        t = t0
        dt = kwargs.get('dt',0.1)
        while (x <= A):
            x += self.F(x,t) * dt + np.sqrt(2*self.D0*dt)*np.random.normal(0.0,1.0)
            t += dt
        return t

    def escape_time_plotpdf(self,x0,t0,A,ntraj):
        samples = [self.escape_time(x0,t0,A) for k in xrange(ntraj)]
        fig = plt.figure()
        ax = plt.axes()
        ax.set_xlabel('$t_0$')
        ax.set_ylabel('$p(t_0)$')
        plt.grid()
        hist, rc = np.histogram(samples,bins='doane',density=True)
        rc = rc[:-1] + 0.5*(rc[1]-rc[0])
        pdf_line, = ax.plot(rc,hist,linewidth=2)
        plt.show()
        
