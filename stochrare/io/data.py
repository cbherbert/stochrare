import numpy as np
import os, pickle, warnings
import scipy.integrate as integrate
from scipy.interpolate import interp1d
from .. import edpy

class Database(dict):
    """ A simple generic database class I use to cache the results from computations on simple stochastic systems to avoid making the same computations over and over again.
    Right now this is just a dictionary with automatic io, using pickle, for disk storage.
    In the long run, there may be better structures to represent the data, such as dbm, pandas or xarray, or even sqlite3. """

    def __init__(self, path):
        """ Automatically read the database on disk if it exists when the object is created """
        self.path = os.path.expanduser(path)
        path = self.path
        try:
            with open(path, 'rb') as f:
                db = pickle.load(f)
                path = db.path
                self.update(db)
        except IOError:
            pass
        if self.path != path:
            warnings.warn("The path saved in the database ("+path+") differs from its actual location ("+self.path+"). It might have been moved manually. The old path will be overwritten if you modify the database. Please check that this is the file you intended to use.", ImportWarning)

    def __setitem__(self, key, value):
        """
        When adding the result of a computation to the database, automatically save it on disk
        """
        dict.__setitem__(self, key, value)
        self.save()

    def __missing__(self, key):
        """
        When requesting parameters for which no realization is stored in the database,
        we simply return an empty array
        """
        return np.array([])

    def save(self):
        """ Save the object on disk"""
        with open(self.path, 'wb') as f:
            pickle.dump(self, f)

    def purge(self):
        """ Delete the database stored on disk """
        try:
            os.remove(self.path)
        except:
            pass


class FirstPassageFP(Database):
    """
    Specializing the above class for storing results
    from numerical integration of the adjoint FP equation.
    Keys are of the form (eps,x0,M).
    So far, I am NOT storing the computational parameters used to solve the FP equation numerically.
    Items correspond to (t,G) where the function G(x0,t) is the probability that a particle
    initially at x0 has not exited yet.
    """
    def __init__(self, path="fpadj.db"):
        Database.__init__(self, path)

    def get_avg(self, eps, M, x0, **kwargs):
        """
        Compute and return the mean first passage time based
        on the solution of the adjoint FP equation stored
        """
        t0 = kwargs.pop('t0', 0.0)
        n = kwargs.pop('order', 1)
        interp = kwargs.pop('interpolate', True)
        method = kwargs.pop('method', 'cdf')
        t, G = self.__getitem__((eps, x0, M))
        if method == 'cdf':
            if interp:
                logG = interp1d(t, np.log(G), fill_value="extrapolate")
                return t0**n+n*integrate.quad(lambda x: x**(n-1)*np.exp(logG(x)), t0, np.inf)[0]
            else:
                return t0**n+n*integrate.trapz(t**(n-1)*G, t)
        else:
            t, pdf = (t[1:-1], -edpy.CenteredFD(t).grad(G))
            return integrate.trapz(t**n*pdf, t)

    def get_pdf(self, eps, M, x0, **kwargs):
        """ Compute and return the PDF of the first passage time based on the stored G """
        time, G = self.__getitem__((eps, x0, M))
        t, pdf = (time[1:-1], -edpy.CenteredFD(time).grad(G))
        if kwargs.get('standardize', False):
            avg = self.get_avg(eps, M, x0, order=1, **kwargs)
            std = self.get_avg(eps, M, x0, order=2, **kwargs)
            std = np.sqrt(std-avg**2)
            return (t-avg)/std, std*pdf
        else:
            return t, pdf

class FirstPassageData(Database):
    """
    Specializing the above class for our specific use case:
    computing and storing first passage time realizations for stochastic models.
    In particular the keys are specialized for our physical/numerical parameters.
    It is very possible that this class breaks many features of the dictionary class.
    Use at your own risks...
    """

    def __init__(self, model, path="~/data/stochtrans/tau.db"):
        self.path = os.path.expanduser(path)
        path = self.path
        try:
            with open(path, 'rb') as f:
                db = pickle.load(f)
                path = db.path
                self.model = db.model
                self.update(db)
        except IOError:
            self.model = model
            pass
        if self.path != path:
            warnings.warn("The path saved in the database ("+path+") differs from its actual location ("+self.path+"). It might have been moved manually. The old path will be overwritten if you modify the database. Please check that this is the file you intended to use.", ImportWarning)
        if self.model != model:
            raise ImportWarning("The model used to generate the database you are trying to access apparently differs from the one you are requesting. Please check. ")


    def __getitem__(self, key):
        """
        We assume that keys are tuples of the form (epsilon,t0,x0,dt,M,n)
        where n is the number of samples we want.
        The underlying dict does NOT use n as part of the key.
        """
        if len(key) != 6:
            return Database.__getitem__(self, key)
        else:
            eps, t0, x0, dt, M, n = key
            data = Database.__getitem__(self, key[:-1])
            if len(data) < n:
                data_new = self.model(eps).escapetime_sample(x0, t0, M, dt=dt, ntraj=n-len(data))
                data = np.concatenate((data, data_new))
                self.__setitem__(key[:-1], data)

        return data[:n]

    def getsamples(self, eps, M, **kwargs):
        """
        A more flexible way to access the database: only epsilon and M are required arguments.
        Initial conditions and timestep are optional arguments.
        By default, we select the earliest available initial time,
        an initial condition on the attractor, and the smallest timestep available.
        Note that the order matters: we select the timestep after the initial time.
        """
        def parseopt(data, optlabel, optdict):
            return (data == optdict.get(optlabel) if optlabel in optdict else True)
        epslist, t0list, x0list, dtlist, Mlist = np.array(zip(*self.keys()))
        mask = (epslist == eps)*(Mlist == M)*parseopt(t0list, 't0', kwargs)*parseopt(x0list, 'x0', kwargs)*parseopt(dtlist, 'dt', kwargs)

        # If t0 is not given, take the largest one in the database
        if 't0' in kwargs:
            t0 = kwargs.get('t0')
        else:
            if len(t0list[mask]) > 0:
                t0 = np.min(t0list[mask])
            else:
                return np.array([])
        # If x0 is not given, assume -sqrt(-t0) (i.e. on the attractor)
        x0 = kwargs.get('x0', -np.sqrt(np.abs(t0)))
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
            return self.__getitem__((eps, t0, x0, dt, M, kwargs.get('n')))
        else:
            return self.__getitem__((eps, t0, x0, dt, M))

    def show_eps(self):
        """ Return the list of noise amplitudes stored in the database """
        if self == {}:
            return set()
        else:
            return set(zip(*self.keys())[0])

    def filter_keys(self, **kwargs):
        """ A flexible way to query the keys stored in the database """
        def parseopt(data, optlabel, optdict):
            return (data == optdict.get(optlabel) if optlabel in optdict else True)
        epslist, t0list, x0list, dtlist, Mlist = np.array(zip(*self.keys()))
        mask = parseopt(epslist, 'eps', kwargs)*parseopt(Mlist, 'M', kwargs)*parseopt(t0list, 't0', kwargs)*parseopt(x0list, 'x0', kwargs)*parseopt(dtlist, 'dt', kwargs)
        return zip(epslist[mask], t0list[mask], x0list[mask], dtlist[mask], Mlist[mask])

class TrajectoryData(Database):
    """
    Specialized database for storing realizations of reacting trajectories,
    indexed by their first-passage time.
    """

    def __init__(self, model, path="~/data/stochtrans/trajs.db"):
        self.path = os.path.expanduser(path)
        path = self.path
        try:
            with open(path, 'rb') as f:
                db = pickle.load(f)
                path = db.path
                self.model = db.model
                self.update(db)
        except IOError:
            self.model = model
            pass
        if self.path != path:
            warnings.warn("The path saved in the database ("+path+") differs from its actual location ("+self.path+"). It might have been moved manually. The old path will be overwritten if you modify the database. Please check that this is the file you intended to use.", ImportWarning)
        if self.model != model:
            raise ImportWarning("The model used to generate the database you are trying to access apparently differs from the one you are requesting. Please check. ")

    def show_eps(self):
        """ Return the list of noise amplitudes stored in the database """
        if self == {}:
            return set()
        else:
            return set(zip(*self.keys())[0])
