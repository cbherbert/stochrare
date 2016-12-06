import numpy as np
import os, pickle
import stochtrans1D

class Database(dict):
    """ The database I use as a cache for first passage time realizations to avoid making the same computations over and over again.
    Right now this is just a dictionary with automatic io, using pickle, for disk storage. 
    In the long run, there may be better structures to represent the data, such as dbm, pandas or xarray, or even sqlite3. """
    
    def __init__(self,path):
        """ Automatically read the database on disk if it exists when the object is created """
        self.path = path
        try :
            with open(self.path,'rb') as f:
                self.update(pickle.load(f))
        except :
            pass

    def __setitem__(self,key,value):
        """ When adding the result of a computation to the database, automatically save it on disk """
        dict.__setitem__(self,key,value)
        with open(self.path,'wb') as f:
            pickle.dump(self,f)

    def __missing__(self,key):
        """ When requesting parameters for which no realization is stored in the database, we simply return an empty array """
        return np.array([])                    

    def purge(self):
        """ Delete the database stored on disk """
        try:
            os.remove(self.path)
        except:
            pass

class FirstPassageData(Database):
    """ Specializing the above class for our specific use case: computing and storing first passage time for stochastic models.
    In particular the keys are specialized for our physical/numerical parameters. 
    It is very possible that this class breaks many features of the dictionary class. Use at your own risks... """

    def __init__(self,path="~/data/stochtrans/tau.db"):
        self.path = os.path.expanduser(path)        
        Database.__init__(self,self.path)

    def __getitem__(self,key):
        """ We assume that keys are tuples of the form (epsilon,t0,x0,dt,M,n) where n is the number of samples we want.
        The underlying dict does NOT use n as part of the key. """
        if len(key) != 6:
            return Database.__getitem__(self,key)
        else:
            eps,t0,x0,dt,M, n = key
            data = Database.__getitem__(self,key[:-1])
            if len(data) < n:
                data_new = stochtrans1D.StochSaddleNode(eps).escapetime_sample(x0,t0,M,dt=dt,ntraj=n-len(data))
                data = np.concatenate((data,data_new))
                self.__setitem__(key[:-1],data)

        return data[:n]

    def getsamples(self,eps,M,**kwargs):
        """ A more flexible way to access the database: only epsilon and M are required arguments.
        Initial conditions and timestep are optional arguments.
        By default, we select the earliest available initial time, an initial condition on the attractor, and the smallest timestep available.
        Note that the order matters: we select the timestep after the initial time. """
        def parseopt(data,optlabel,optdict):
            return (data == optdict.get(optlabel) if optlabel in optdict else True)
        epslist,t0list,x0list,dtlist,Mlist = np.array(zip(*self.keys()))
        mask = (epslist==eps)*(Mlist == M)*parseopt(t0list,'t0',kwargs)*parseopt(x0list,'x0',kwargs)*parseopt(dtlist,'dt',kwargs)

        # If t0 is not given, take the largest one in the database
        if 't0' in kwargs:
            t0 = kwargs.get('t0')
        else:
            if len(t0list[mask]) >0:
                t0 = np.min(t0list[mask])
            else:
                return np.array([])
        # If x0 is not given, assume -sqrt(-t0) (i.e. on the attractor)
        x0 = kwargs.get('x0',-np.sqrt(np.abs(t0)))
        # If dt is not given, take the smallest value in the database
        if 'dt' in kwargs:
            dt = kwargs.get('dt')
        else:
            if len(dtlist[mask*(t0list == t0)]) > 0:
                dt = np.min(dtlist[mask*(t0list == t0)])
            else:
                return np.array([])

        # If n is not given, read all the samples we have
        if 'n' in kwargs:
            return self.__getitem__((eps,t0,x0,dt,M,kwargs.get('n')))
        else:
            return self.__getitem__((eps,t0,x0,dt,M))
    
    def show_eps(self):
        """ Return the list of noise amplitudes stored in the database """
        if self == {}:
            return set()
        else:
            return set(zip(*self.keys())[0])
    
    
