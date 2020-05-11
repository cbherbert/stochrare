"""
Code for benchmarking some core features of stochrare.
This provides a basis for comparing the performance of different releases of the code.
"""
from os import path
import glob
import timeit
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import numpy as np
from numba import jit, jitclass, float32
from stochrare.dynamics.diffusion1d import DiffusionProcess1D, ConstantDiffusionProcess1D, OrnsteinUhlenbeck1D
from stochrare.dynamics.diffusion import OrnsteinUhlenbeck

def runnb(notebook_filename):
    ep = ExecutePreprocessor(timeout=3600, kernel_name='python3')
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
    ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})
    return nb
#with open('executed_notebook.ipynb', 'w', encoding='utf-8') as f:
#    nbformat.write(nb, f)

def oup_vanilla(niter, dt):
    np.random.seed(100)
    x = [0.0, ]
    for _ in range(1, niter):
        x += [(1-dt)*x[-1]+np.sqrt(0.2)*np.random.normal(0.0, np.sqrt(dt))]
    return np.array(x)

def oup_vanilla2(niter, dt):
    np.random.seed(100)
    x = [0.0, ]
    wiener = np.random.normal(0.0, np.sqrt(dt), niter-1)
    for w in wiener:
        x += [(1-dt)*x[-1]+np.sqrt(0.2)*w]
    return np.array(x)

def oup_vanilla3(niter, dt):
    np.random.seed(100)
    x = np.zeros(niter)
    wiener = np.random.normal(0.0, np.sqrt(dt), niter-1)
    for i, w in enumerate(wiener, 1):
        x[i] = (1-dt)*x[i-1]+np.sqrt(0.2)*w
    return x

@jit(nopython=True)
def oup_numba(niter, dt):
    np.random.seed(100)
    x = [0.0, ]
    for _ in range(1, niter):
        x += [(1-dt)*x[-1]+np.sqrt(0.2)*np.random.normal(0.0, np.sqrt(dt))]
    return np.array(x)

@jit(nopython=True)
def oup_numba2(niter, dt):
    np.random.seed(100)
    x = [0.0, ]
    wiener = np.random.normal(0.0, np.sqrt(dt), niter-1)
    for w in wiener:
        x += [(1-dt)*x[-1]+np.sqrt(0.2)*w]
    return np.array(x)

@jit(nopython=True)
def oup_numba3(niter, dt):
    np.random.seed(100)
    x = np.zeros(niter)
    wiener = np.random.normal(0.0, np.sqrt(dt), niter-1)
    for i, w in enumerate(wiener, 1):
        x[i] = (1-dt)*x[i-1]+np.sqrt(0.2)*w
    return x

@jit(nopython=True)
def update(xn, wn, dt):
    return (1-dt)*xn+np.sqrt(0.2)*wn

@jit(nopython=True)
def oup_numba4(niter, dt):
    np.random.seed(100)
    x = np.zeros(niter)
    wiener = np.random.normal(0.0, np.sqrt(dt), niter-1)
    for i, w in enumerate(wiener, 1):
        x[i] = update(x[i-1], w, dt)
    return x

@jit(nopython=True)
def oup_numba5(niter, dt, fun):
    np.random.seed(100)
    x = np.zeros(niter)
    wiener = np.random.normal(0.0, np.sqrt(dt), niter-1)
    for i, w in enumerate(wiener, 1):
        x[i] = (1-dt)*fun(x[i-1])+np.sqrt(0.2)*w
    return x

idf = jit(lambda x: x, nopython=True)

@jitclass([('D', float32)])
class OupNumba:
    """
    Numba with OOP, option 1: jitclass decorator
    """
    def __init__(self, D):
        self.D = D

    def trajectory(self, niter, dt):
        np.random.seed(100)
        x = np.zeros(niter)
        wiener = np.random.normal(0.0, np.sqrt(dt), niter-1)
        for i, w in enumerate(wiener, 1):
            x[i] = (1-dt)*x[i-1]+np.sqrt(2*self.D)*w
        return x

class OupNumba2:
    """
    Numba with OOP, option 2: wrapper
    """
    def trajectory(self, niter, dt):
        return oup_numba3(niter, dt)

class OupNumba3:
    """
    Numba with OOP, option 3: static method
    """
    @staticmethod
    @jit(nopython=True)
    def trajectory(niter, dt):
        np.random.seed(100)
        x = np.zeros(niter)
        wiener = np.random.normal(0.0, np.sqrt(dt), niter-1)
        for i, w in enumerate(wiener, 1):
            x[i] = (1-dt)*x[i-1]+np.sqrt(0.2)*w
        return x

class OupNumba3b:
    """
    Numba with OOP, option 3b: static method with np.dot
    """
    @staticmethod
    @jit(nopython=True)
    def trajectory(niter, dt):
        np.random.seed(100)
        x = np.zeros((niter, 1))
        wiener = np.random.normal(0.0, np.sqrt(dt), niter-1)
        wiener = wiener.reshape(niter-1, 1)
        diff=np.array([np.sqrt(2)])
        for i, w in enumerate(wiener, 1):
            x[i][0] = (1-dt)*x[i-1][0]+np.dot(diff, w)
        return x


class OupNumba4:
    """
    Numba with OOP with function member
    """
    def __init__(self, D):
        self.D = D
        self.fun = lambda x: x
        self.jitfun = jit(lambda x: x, nopython=True)

    @staticmethod
    @jit(nopython=True)
    def _trajectory_static(niter, dt, D, fun):
        np.random.seed(100)
        x = np.zeros(niter)
        wiener = np.random.normal(0.0, np.sqrt(dt), niter-1)
        for i, w in enumerate(wiener, 1):
            x[i] = x[i-1]-dt*fun(x[i-1])+np.sqrt(2*D)*w
        return x

    def trajectory(self, niter, dt):
        #jitfun = jit(self.fun, nopython=True)
        return self._trajectory_static(niter, dt, self.D, self.jitfun)

class OupNumba4b:
    """
    Numba with OOP with function member, variant
    """
    def __init__(self, D):
        self.D = D
        self.fun = lambda x: x
        self.jitfun = jit(lambda x: x, nopython=True)

    @staticmethod
    @jit(nopython=True)
    def _trajectory_static(x, wiener, dt, D, fun):
        index = 1
        for w in wiener:
            xn = x[index-1]
            x[index] = xn -fun(xn)*dt + np.sqrt(2*D)*w
            index = index + 1
        return x

    def trajectory(self, niter, dt):
        np.random.seed(100)
        x = np.zeros(niter)
        wiener = np.random.normal(0.0, np.sqrt(dt), niter-1)
        return self._trajectory_static(x, wiener, dt, self.D, self.jitfun)


def benchmark_notebooks():
    """
    Run the demo notebooks and measure the time needed.
    """
    print('Running notebooks in doc/notebooks')
    for notebook in glob.glob('doc/notebooks/*.ipynb'):
        duration = timeit.timeit(f'runnb("{notebook}")', 'from __main__ import runnb', number=1)
        print(f"Notebook: {path.basename(notebook)} - Elapsed time: {duration}s")

def benchmark_trajectory(nb=100):
    """
    Measure the time needed for numerical integration of SDEs with stochrare.
    """
    oup_euler = DiffusionProcess1D(lambda x, t: -x, lambda x, t: np.sqrt(0.2))
    oup_euler_constant = ConstantDiffusionProcess1D(lambda x, t: -x, 0.1)
    oup_gillespie = OrnsteinUhlenbeck1D(0, 1, 0.1)
    oup_euler.trajectory(0, 0, dt=0.01)
    oup_euler_constant.trajectory(0, 0, dt=0.01)
    oup_gillespie.trajectory(0, 0, dt=0.01, method='gillespie')
    duration = timeit.timeit(lambda: oup_euler.trajectory(0, 0, dt=0.01), number=nb)*1000
    print(f"Solving SDE (Euler-Maruyama DiffusionProcess1D)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    duration = timeit.timeit(lambda: oup_euler_constant.trajectory(0, 0, dt=0.01), number=nb)*1000
    print(f"Solving SDE (Euler-Maruyama ConstantDiffusionProcess1D)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    duration = timeit.timeit(lambda: oup_gillespie.trajectory(0, 0, dt=0.01, method='gillespie'),
                             number=nb)*1000
    print(f"Solving SDE (Gillespie)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    oup2d = OrnsteinUhlenbeck(0, 1, 0.1, 2)
    oup2d.trajectory(np.array([0, 0]), 0, dt=0.01)
    duration = timeit.timeit(lambda: oup2d.trajectory(np.array([0, 0]), 0, dt=0.01), number=nb)*1000
    print(f"Solving 2D SDE (DiffusionProcess)... {nb} realizations (1000 samples) in {duration:.3f}ms")


def benchmark_trajectory_vanilla(nb=100):
    """
    Measure the time needed for numerical integration of SDEs with a naive loop.
    """
    duration = timeit.timeit('oup_vanilla(1000, 0.01)', number=nb, globals=globals())*1000
    print(f"Solving SDE (vanilla 1)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    duration = timeit.timeit('oup_vanilla2(1000, 0.01)', number=nb, globals=globals())*1000
    print(f"Solving SDE (vanilla 2)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    duration = timeit.timeit('oup_vanilla3(1000, 0.01)', number=nb, globals=globals())*1000
    print(f"Solving SDE (vanilla 3)... {nb} realizations (1000 samples) in {duration:.3f}ms")

def benchmark_trajectory_numba(nb=100):
    """
    Measure the time needed for numerical integration of SDEs with numba.
    """
    oupnumba4_oop = OupNumba4(0.1)
    oupnumba4b_oop = OupNumba4b(0.1)
    oup_numba(1000, 0.01)
    oup_numba2(1000, 0.01)
    oup_numba3(1000, 0.01)
    oup_numba4(1000, 0.01)
    oup_numba5(1000, 0.01, idf)
    OupNumba(0.1).trajectory(1000, 0.01)
    OupNumba2().trajectory(1000, 0.01)
    OupNumba3().trajectory(1000, 0.01)
    oupnumba4_oop.trajectory(1000, 0.01)
    oupnumba4b_oop.trajectory(1000, 0.01)
    #OupNumba4._trajectory_static(1000, 0.01, 0.1, idf)
    duration = timeit.timeit('oup_numba(1000, 0.01)', number=nb, globals=globals())*1000
    print(f"Solving SDE (numba 1)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    duration = timeit.timeit('oup_numba2(1000, 0.01)', number=nb, globals=globals())*1000
    print(f"Solving SDE (numba 2)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    duration = timeit.timeit('oup_numba3(1000, 0.01)', number=nb, globals=globals())*1000
    print(f"Solving SDE (numba 3)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    duration = timeit.timeit('oup_numba4(1000, 0.01)', number=nb, globals=globals())*1000
    print(f"Solving SDE (numba 4)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    duration = timeit.timeit('oup_numba5(1000, 0.01, idf)', number=nb, globals=globals())*1000
    print(f"Solving SDE (numba 5)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    duration = timeit.timeit('OupNumba(0.1).trajectory(1000, 0.01)', number=nb,
                             globals=globals())*1000
    print(f"Solving SDE (numba OUP 1)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    duration = timeit.timeit('OupNumba2().trajectory(1000, 0.01)', number=nb,
                             globals=globals())*1000
    print(f"Solving SDE (numba OUP 2)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    duration = timeit.timeit('OupNumba3().trajectory(1000, 0.01)', number=nb,
                             globals=globals())*1000
    print(f"Solving SDE (numba OUP 3)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    duration = timeit.timeit('OupNumba3b().trajectory(1000, 0.01)', number=nb,
                             globals=globals())*1000
    print(f"Solving SDE (numba OUP 3b)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    duration = timeit.timeit(lambda: oupnumba4_oop.trajectory(1000, 0.01), number=nb)*1000
    print(f"Solving SDE (numba OUP 4)... {nb} realizations (1000 samples) in {duration:.3f}ms")
    duration = timeit.timeit(lambda: oupnumba4b_oop.trajectory(1000, 0.01), number=nb)*1000
    print(f"Solving SDE (numba OUP 4b)... {nb} realizations (1000 samples) in {duration:.3f}ms")

def benchmark_fokkerplanck():
    """
    Measure the time needed for numerical solution of the Fokker-Planck equation.
    """
    stmt = "FokkerPlanck1D(lambda x, t: 0, lambda x, t: 1).fpintegrate(0, 10, dt=0.001, npts=200)"
    duration = timeit.timeit(stmt, 'from stochrare.fokkerplanck import FokkerPlanck1D', number=10)
    print(f"Solving Fokker-Planck equation: explicit (Euler) method (10000 time steps, 200 points, 10 repetitions) in {duration}s")
    stmt = "FokkerPlanck1D(lambda x, t: 0, lambda x, t: 1).fpintegrate(0, 10, dt=0.02, npts=200, method='implicit')"
    duration = timeit.timeit(stmt, 'from stochrare.fokkerplanck import FokkerPlanck1D', number=10)
    print(f"Solving Fokker-Planck equation: implicit method (500 time steps, 200 points, 10 repetitions) in {duration}s")

if __name__ == '__main__':
    benchmark_notebooks()
    benchmark_trajectory()
    benchmark_fokkerplanck()
