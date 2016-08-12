import random
import numpy as np
import matplotlib.pyplot as plt

class DynSaddleNode:

    def __init__(self,Damp):
        self.D0 = Damp # Amplitude of the diffusion term (noise)
        
    def gradV(self,x):
        return -x**2

    def trajectory(self,x0,t0,**kwargs):
        """ Integrate a trajectory with given initial condition (t0,x0), with time step dt, for niter iterations """
        x      = [x0]
        dt     = kwargs.get('dt',0.1)
        time   = kwargs.get('T',10)
        tarray = np.linspace(t0,t0+time,num=time/dt+1)
        for t in tarray[1:]:    
            x += [ x[-1] + (t- self.gradV(x[-1])) * dt + np.sqrt(2*self.D0*dt)*random.gauss(0.0,1.0)]
        return tarray,np.array(x)

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
            x += (t- self.gradV(x)) * dt + np.sqrt(2*self.D0*dt)*random.gauss(0.0,1.0)
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
