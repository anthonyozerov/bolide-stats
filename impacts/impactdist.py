from tqdm import tqdm
from astropy.coordinates import SkyCoord, HeliocentricTrueEcliptic
from astropy.coordinates import ICRS, ITRS
import astropy.units as u
from astropy.coordinates import get_sun
from astropy.time import Time
from bolides.astro_utils import get_solarhour, get_sun_alt
from datetime import datetime, timedelta
from math import pi
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os


def rotateVector(vect, axis, theta):
    """ Rotate vector around the given axis by a given angle.
        From WesternMeteorPyLib (Denis Vida, MIT License)"""
    rot_M = scipy.linalg.expm(np.cross(np.eye(3), axis/np.linalg.norm(axis, 2)*theta))

    return np.dot(rot_M, vect)


def sample_impacts(min_vinf=None, max_vinf=None,
                   min_eclon=None, max_eclon=None,
                   min_eclat=None, max_eclat=None,
                   vinf=None, eclon=None, eclat=None,
                   n=1000, dt=None):

    if min_vinf is None:
        min_vinf = 0
    if min_eclon is None:
        min_eclon = -180
    if min_eclat is None:
        min_eclat = -90
    if max_vinf is None:
        max_vinf = 100
    if max_eclon is None:
        max_eclon = 180
    if max_eclat is None:
        max_eclat = 90

    if any([vinf is None, eclon is None, eclat is None]):
        data = np.loadtxt('../matlab/pvil.dat', skiprows=0).reshape((360, 180, 100))
        prob = data/np.sum(data)
        prob = prob[(min_eclon+180):(max_eclon+180), (min_eclat+90):(max_eclat+90), min_vinf:max_vinf]
        print(prob.shape)
        prob = prob/np.sum(prob)
        idx = np.random.choice(np.arange(0, np.product(prob.shape)), replace=True, p=prob.flatten(), size=n)
        eclons, eclats, vinfs = np.unravel_index(idx, prob.shape)
        eclons = eclons+min_eclon+0.5
        eclats = eclats+min_eclat+0.5
        vinfs = vinfs+min_vinf+0.5
    # plt.hist(eclats, weights=1/np.cos(np.radians(eclats)))
    # plt.show()

    if vinf is not None:
        print('vinf fixed')
        assert min_vinf <= vinf and vinf <= max_vinf
        vinfs = np.full(n, vinf)
    if eclon is not None:
        print('eclon fixed')
        assert min_eclon <= eclon and eclon <= max_eclon
        eclons = np.full(n, eclon)
    if eclat is not None:
        print('eclat fixed')
        assert min_eclat <= eclat and eclat <= max_eclat
        eclats = np.full(n, eclat)

    d1 = datetime.fromisoformat('2018-01-01')
    dts = d1 + np.array([timedelta(seconds=s) for s in (np.random.random(n) * 60*60*24*365.25*5)])
    if dt is not None:
        print('dt fixed')
        if hasattr(dt, '__iter__'):
            assert len(dt) == n
        else:
            dt = np.full(dts.shape, dt)

    coords = get_sun(Time(dts))
    sollons = [c.ra.deg for c in coords]
    sollons = np.radians(sollons)

    mu = 398600  # km3 s-2 standard gravitational parameter of earth (gm)
    r = 6371  # km mean radius
    vesc = np.sqrt(2*mu/r)  # force minimum impact speed to be escape velocity

    vr = np.sqrt(vinfs**2 + vesc**2)  # km s-1 speed when impacts earth
    bmax = r*vr/vinfs  # km max impact parameter
    b = np.sqrt(np.random.uniform(size=n)*(bmax**2))  # km random radius within max

    a = mu/(vinfs**2)  # km semi-major axis of trajectory relative to earth
    e = np.sqrt(1 + (b/a)**2)  # eccentricity of hyperbola
    f = np.arccos(((b**2)/a/r - 1)/e)  # rad  true anomaly
    theta = np.arccos(1/e)  # half-angle of asymptotes
    psi = pi - f - theta
    Z = np.sin(psi)  # in planet radii
    # impact point on sphere
    th = np.random.uniform(size=n, low=-pi, high=pi)  # angle of orientation of impact parameter
    y = Z*np.cos(th)
    z = Z*np.sin(th)  # impact parameters in planet radii
    x = np.sqrt(1-(y**2+z**2))
    X = np.vstack([x, y, z]).T  # r=1

    axis = np.array([0, 1, 0])
    X = [rotateVector(x, axis, -np.radians(eclat)) for x, eclat in tqdm(zip(X, eclats))]
    axis = np.array([0, 0, 1])
    X = [rotateVector(x, axis, -np.radians(eclon)+sollon-pi/2) for x, eclon, sollon in tqdm(zip(X, eclons, sollons))]

    X = np.array(X)
    coords = SkyCoord(HeliocentricTrueEcliptic(x=X[:, 0]*1000*u.pc,
                                               y=X[:, 1]*1000*u.pc,
                                               z=X[:, 2]*1000*u.pc, representation_type='cartesian'))

    icrs = coords.transform_to(ICRS)
    icrs.obstime = dts

    itrs = icrs.transform_to(ITRS)
    locs = np.vstack((itrs.x.value, itrs.y.value, itrs.z.value)).T
    lats = np.degrees(np.arctan(locs[:, 2]/np.linalg.norm(locs[:, [0, 1]], axis=1)))
    lons = np.degrees(np.arctan2(locs[:, 1], locs[:, 0]))

    from bolides.fov_utils import get_boundary
    from shapely.geometry import Point
    goes_e, goes_w_i, goes_w_ni = get_boundary(['goes-e', 'goes-w-i', 'goes-w-ni'], crs='epsg:4326')
    within = [Point([lon, lat]).within(goes_e) for lon, lat in tqdm(zip(lons, lats))]

    solarhours = np.array([get_solarhour(d, l) for d, l in tqdm(zip(dts, lons))])

    sun_alts = np.array([get_sun_alt(d, lat, lon)[1] for d, lat, lon in tqdm(zip(dts, lats, lons))])

    result = {'lats': lats, 'solarhours': solarhours, 'sun_alts': sun_alts,
              'lats_glm': lats[within], 'solarhours_glm': solarhours[within], 'sun_alts_glm': sun_alts[within]}

    return result


def get_densities(df, samples, colname):
    for suffix in ['', '_glm']:
        lats = samples['lats'+suffix]
        solarhours = samples['solarhours'+suffix]
        sun_alts = samples['sun_alts'+suffix]

        # compute solar hour density
        solarhours = np.array(list(solarhours) + list(solarhours+24) + list(solarhours-24))
        X = solarhours[:, np.newaxis]
        kde = KernelDensity(kernel="gaussian", bandwidth=1).fit(X)
        x_plot = np.array(df['x_solarhour'])
        density = np.exp(kde.score_samples(x_plot[:, np.newaxis]))
        # multiply by 3 to take into account the data triplication that occurs above
        # to get the KDE to be circular
        df[colname+'_solarhour'+suffix] = density*3

        X = sun_alts[:, np.newaxis]
        kde = KernelDensity(kernel="gaussian", bandwidth=4).fit(X)
        x_plot = np.array(df['x_sun_alt'])
        density = np.exp(kde.score_samples(x_plot[:, np.newaxis]))
        df[colname+'_sun_alt'+suffix] = density

        X = lats[:, np.newaxis]
        kde = KernelDensity(kernel="gaussian", bandwidth=4).fit(X, sample_weight=1/np.cos(np.radians(lats)))
        x_plot = np.array(df['x_lat'])
        density = np.exp(kde.score_samples(x_plot[:, np.newaxis]))
        df[colname+'_lat'+suffix] = density

    return df

if not os.path.exists('impact-dists.csv'):
    df = pd.DataFrame()
    df['x_solarhour'] = np.linspace(0, 24, 200)
    df['x_sun_alt'] = np.linspace(-90, 90, 200)
    df['x_lat'] = np.linspace(-90, 90, 200)
    for min_vinf in [0, 45, 50, 65]:
        print(f'Computing impact distributions for vinf >= {min_vinf}')
        colname = f'v{min_vinf}'
        # n should be 200000 for good stats
        samples = sample_impacts(min_vinf=min_vinf, n=400000)
        df = get_densities(df, samples, colname)

    df.to_csv('impact-dists.csv', index=False)

if not os.path.exists('leonids-dists.csv'):
    n = 400000
    df = pd.DataFrame()
    df['x_solarhour'] = np.linspace(0, 24, 200)
    df['x_sun_alt'] = np.linspace(-90, 90, 200)
    df['x_lat'] = np.linspace(-90, 90, 200)
    d1 = datetime.fromisoformat('2018-11-15')
    dts = d1 + np.array([timedelta(seconds=s) for s in (np.random.random(n) * 60*60*24*5)])
    icrs = SkyCoord(ICRS(ra=154*u.deg, dec=21.8*u.deg, distance=1000*u.pc))
    helio = icrs.transform_to(HeliocentricTrueEcliptic)
    eclat = helio.lat.value
    eclon = helio.lon.value
    eclon -= 235 - 90
    samples = sample_impacts(n=n, dt=dts, eclat=eclat, eclon=eclon, vinf=70.2)
    df = get_densities(df, samples, colname='')
    df.to_csv('leonids-dists.csv', index=False)

if not os.path.exists('perseids-dists.csv'):
    n = 400000
    df = pd.DataFrame()
    df['x_solarhour'] = np.linspace(0, 24, 200)
    df['x_sun_alt'] = np.linspace(-90, 90, 200)
    df['x_lat'] = np.linspace(-90, 90, 200)
    d1 = datetime.fromisoformat('2018-08-10')
    dts = d1 + np.array([timedelta(seconds=s) for s in (np.random.random(n) * 60*60*24*5)])
    icrs = SkyCoord(ICRS(ra=48.2*u.deg, dec=58.1*u.deg, distance=1000*u.pc))
    helio = icrs.transform_to(HeliocentricTrueEcliptic)
    eclat = helio.lat.value
    eclon = helio.lon.value
    eclon -= 140 - 90
    samples = sample_impacts(n=n, dt=dts, eclat=eclat, eclon=eclon, vinf=59.1)
    df = get_densities(df, samples, colname='')
    df.to_csv('perseids-dists.csv', index=False)
