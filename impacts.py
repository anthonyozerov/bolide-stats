import matplotlib.pyplot as plt
from oct2py import octave
from tqdm import tqdm
from math import radians
import numpy as np
from sklearn.neighbors import KernelDensity

octave.eval('pkg load statistics')
octave.addpath('./matlab')
octave.run('impact-test.m')
#octave.run('getdist.m')

# arguments in degrees. Vinf in km/s
def shower_impact_dist(name, n, Vinf, inclination, eclipticlon):
    lats = np.degrees(octave.get_lat(n, Vinf, radians(inclination), radians(eclipticlon)).flatten())
    X = lats[:, np.newaxis]
    kde = KernelDensity(kernel="gaussian", bandwidth=4).fit(X, sample_weight=1/np.cos(np.radians(lats)))
    x_plot = np.linspace(-55, 55, 200)
    density = np.exp(kde.score_samples(x_plot[:,np.newaxis]))
    density /= density[int(len(density)/2)] #  normalize by midpoint
    data = np.array(list(zip(x_plot, density)))
    np.savetxt(f'impacts/{name}-dist.csv', data, delimiter=',', header="lat, dens")

def get_eclipticlat(vinf, pdf):
    return np.random.choice(np.arange(0,90),p=pdf[:,vinf]/sum(pdf[:,vinf])) * np.random.choice([-1,1])

def vinf_impact_dist(vinfs, n):

    pdf = np.loadtxt('matlab/pvi2018.dat')
    pdf = np.flipud(pdf[:90,:] + np.flipud(pdf[90:,:]))

    x_plot = np.linspace(-55, 55, 200)
    data = np.zeros((len(x_plot),len(vinfs)+1))
    data[:,0] = x_plot
    for i, vinf in enumerate(vinfs):
        vs = [vinf] * n
        eclipticlats = [get_eclipticlat(v, pdf) for v in vs]
        lats = np.degrees(octave.getdist(vs,np.radians(eclipticlats)).flatten())
        X = lats[:, np.newaxis]
        kde = KernelDensity(kernel="gaussian", bandwidth=4).fit(X, sample_weight=1/np.cos(np.radians(lats)))
        density = np.exp(kde.score_samples(x_plot[:,np.newaxis]))
        #plt.plot(x_plot, density)
        #plt.show()
        density /= density[int(len(density)/2)]
        data[:,i+1] = density
    np.savetxt('impacts/lat-vel-dist.csv', data, delimiter=',', header="lat, "+", ".join([str(v) for v in vinfs]))

shower_impact_dist('leonids', 100000, 70.66, 162, 148)
vinf_impact_dist([50, 55, 60, 65, 68], 100000)
