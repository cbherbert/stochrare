import random
import numpy as np
import matplotlib.pyplot as plt

# custom numpy running mean function; a panda solution might be more efficient
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N 

class DoubleWell:
    """ Double well potential model, possibly including periodic forcing and noise """

    def __init__(self,Famp,omega,Damp):
        self.F0  = Famp    # Amplitude of the periodic forcing
        self.Om0 = omega   # Frequency of the periodic forcing
        self.D0  = Damp    # Amplitude of the diffusion term (noise) 

    def gradV(self,x):
        return x*(x**2-1)

    def trajectory(self,x0,niter,dt):
        """ Integrate a trajectory with given initial condition x0, with time step dt, for niter iterations """
        x = [x0]
        for t in np.linspace(0,niter*dt,num=niter-1):    
            x += [ x[-1] + ( self.F0*np.sin(self.Om0*t) - self.gradV(x[-1])) * dt + np.sqrt(2*self.D0*dt)*random.gauss(0.0,1.0)]
        return np.array(x)

    def trajectoryplot(self,*args,**kwargs):
        """ Plot previously computed trajectories with initial time t0 and time unit dt """
        dt = kwargs.get('dt',1)
        t0 = kwargs.get('t0',0)        
        for x in args:
            plt.plot(np.linspace(t0,(len(x)-1)*dt,num=len(x)),x)
        plt.xlabel('$t$')
        plt.ylabel('$x(t)$')
        plt.show()
    
    def phaseportrait(self,a,b,ntraj,niter,dt):
        """ Compute and plot the trajectories for an ensemble of initial conditions """
        time = np.linspace(0,(niter-1)*dt,num=niter)
        for x0 in np.linspace(a,b,num=ntraj):
            plt.plot(time,self.trajectory(x0,niter,dt))        
        plt.xlabel('$t$')
        plt.ylabel('$x(t)$')
        plt.show()

    def transitionrate(self,x,**kwargs):
        """ Count the number of transitions from one attractor to the other for a given trajectory """
        window = kwargs.get('avg',1)
        y = running_mean(x,window) if window > 1 else x
        return float((y[1:]*y[:-1] < 0).sum())/len(y)

