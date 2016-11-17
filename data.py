import numpy as np
import os, pickle

class Database(dict):
    """ The database I use as a cache for first passage time realizations to avoid making the same computations over and over again.
    Right now this is just a dictionary with automatic io, using pickle, for disk storage. 
    In the long run, there may be better structures to represent the data, such as dbm, pandas or xarray, or even sqlite3. """

    path = 'tau.db'
    
    def __init__(self):
        """ Automatically read the database on disk if it exists when the object is created """
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
