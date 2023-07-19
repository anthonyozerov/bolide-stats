import numpy as np
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
# from haversine import haversine_vector
import pandas as pd
from global_land_mask import globe
import pymc as pm
from tqdm import tqdm
import pyproj
# import cartopy.crs as ccrs

# import os
# import pickle

from shapely.geometry import Point

from bolides.fov_utils import get_boundary
from bolides import ShowerDataFrame
from bolides.constants import GOES_W_LON

from geo_utils import get_flash_density, get_areas
from partition import random_partition


# function to add lons, lats, distances from nadir, and other
# variables to the dataframe of polygons and counts.
def get_data(gdf, fov_center=None, mapping=None, shower_data=None):

    # get the centroids of the polygons and their latitudes and longitudes
    centroids = [list(poly.centroid.xy) for poly in gdf.geometry]
    xs = [c[0][0] for c in centroids]
    ys = [c[1][0] for c in centroids]
    points = [Point(x, y) for x, y in zip(xs, ys)]
    points = GeoDataFrame(geometry=points, crs=gdf.crs).to_crs('epsg:4326').geometry
    lons = np.array([p.x for p in points])
    lats = np.array([p.y for p in points])

    # obtain a list of coordinates to compute the distances from the center of the field of view
    coords = list(zip(lats, lons))
    if fov_center is not None:
        def unit(vec):
            return vec/np.linalg.norm(vec)
        transformer = pyproj.Transformer.from_crs(
            {"proj": 'latlong', "ellps": 'WGS84', "datum": 'WGS84'},
            {"proj": 'geocent', "ellps": 'WGS84', "datum": 'WGS84'}
        )

        nadir_lat = fov_center[0]
        nadir_lon = fov_center[1]

        goes_alt = 35786 * 1000
        sat_vec = np.array(transformer.transform(nadir_lon, nadir_lat, goes_alt, radians=False))

        # TODO: this isn't quite right because of the differing height of the GLM reference ellipsoid
        loc_vecs = np.array([transformer.transform(lon, lat, 10*1000, radians=False) for lat, lon in coords])

        look_vecs = sat_vec-loc_vecs
        aois = [np.degrees(np.arccos(np.dot(unit(sat_vec), unit(look_vec)))) for look_vec in look_vecs]

    # get the pixel pitches of the pixels on the CCD corresponding to the centroids
    # if mapping is not None:
    #     print('getting pixel pitches')
    #     pitches = np.array([get_pitch(mapping, p.y, p.x) for p in tqdm(points)])
    #     pitches /= 900

    from netCDF4 import Dataset
    flash_data = Dataset('data/LISOTD_HRFC_V2.3.2015.nc')
    flash_dens = np.array([get_flash_density(flash_data, lat, lon) for lat, lon in zip(lats, lons)])
    flash_dens = np.nan_to_num(flash_dens, copy=True, nan=np.nanmean(flash_dens))
    assert sum(np.isnan(flash_dens)) == 0

    cloud_data = np.loadtxt('data/cloud-avg.txt')
    # use negative lat because cloud-avg is indexed differently
    from geo_utils import idx_from_latlon
    idxs = [idx_from_latlon(-lat, lon) for lat, lon in zip(lats, lons)]
    cloud_props = [cloud_data[idx] for idx in idxs]

    # calculate whether a centroid is over land or not
    land = np.array([globe.is_land(lat, lon) for lat, lon in zip(lats, lons)])
    land = land.astype(int)

    if shower_data is not None:
        for s, filename in shower_data.items():
            if s in gdf.columns:
                print(f'loading {filename}')
                shower_rate = pd.read_csv(filename)
                shower_lats = np.array(shower_rate['lat'])
                rates = np.array(shower_rate['dens'])

                gdf[s+'rate'] = gdf[s]*np.interp(lats, shower_lats, rates)

    # add new columns to the dataframe
    gdf['lat'] = lats
    gdf['lon'] = lons
    gdf['aoi'] = aois
    gdf['flash_dens'] = flash_dens
    gdf['cloud_prop'] = cloud_props
    gdf['land'] = land

    print(f'created dataframe with columns {list(gdf.columns)}')

    return gdf


# given dataset, fit the model
def fit(data, f_lat, f_fov, biases, showers, shower_known=False, reg=True, **kwargs):
    # extract variables from the data
    lat = np.array(data['lat'])
    fov = np.array(data['aoi'])
    area = np.array(data['area'])
    max_area = max(area)
    area = area/max_area
    count = np.array(data['count'])
    duration = np.array(data['duration'])

    # create data matrix
    X = pd.DataFrame()
    for bias in biases:
        bdata = np.array(data[bias]).astype(float)
        X[bias] = bdata
    fov_terms = 0
    if len(f_fov) > 0:
        for term in f_fov.split('+'):
            power = int(term.split('^')[1])
            fovpower = fov**power
            X[f'fov{power}'] = fovpower
            fov_terms += 1
    lat_terms = 0
    if len(f_lat) > 0:
        for term in f_lat.split('+'):
            power = int(term.split('^')[1])
            latpower = lat**power
            X[f'lat{power}'] = latpower
            lat_terms += 1
    if not shower_known:
        for s in showers:
            indicator = np.array(data[s])
            for term in f_lat.split('+'):
                power = int(term.split('^')[1])
                latpower = lat**power
                X[f'{s}lat{power}'] = latpower*indicator

    # var_names = X.columns.values

    # standardize X
    mean = X.mean()
    scale = X.std()
    X -= mean
    X /= scale

    # define model
    with pm.Model(coords={"predictors": X.columns.values}):
        import pytensor.tensor as tt

        if reg:
            # https://doi.org/10.1214/17-EJS1337SI
            # https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_overview.html

            # global shrinkage parameter
            tau = pm.HalfCauchy("tau", beta=1)

            # slab width
            c2 = pm.InverseGamma("c2", 2/2, 2/2)  # pg 8

            # local shrinkage parameters
            lam = pm.HalfStudentT("lam", 2, dims="predictors")

            # standard normal Z just for convenience
            z = pm.Normal("z", 0, 1, dims="predictors")

            # for convenience
            lambda_tilde = np.sqrt(c2*lam/(c2 + tau**2 * lam**2))

            # beta ~ N(0, tau^2lambda^2)
            beta = pm.Deterministic("beta", z * tau * lambda_tilde, dims="predictors")

        else:
            beta = pm.Normal("beta", mu=0, sigma=1, dims="predictors")

        n1 = len(biases)+len(f_fov.split('+'))
        bias_term = np.exp(tt.dot(X.values[:, :n1], beta[:n1]))

        if shower_known:
            print('shower rates are known')
            intercept = pm.Cauchy("intercept", alpha=0, beta=0.5)
            shower_rate = 0
            for s in showers:
                s_intercept = pm.HalfCauchy(f"{s}intercept", beta=0.5)
                shower_rate += s_intercept*np.array(data[f'{s}rate'])
            total_rate = np.exp(intercept + tt.dot(X.values[:, n1:], beta[n1:])) + shower_rate
        else:
            total_rate = 0
            ni = n1
            step = len(f_fov.split('+'))
            for i in range(len(showers)+1):
                s = (['']+showers)[i]
                intercept = pm.Cauchy(f"{s}intercept", alpha=0, beta=0.5)
                if i > 0:
                    intercept *= np.array(data[showers[i-1]])
                total_rate += np.exp(intercept + tt.dot(X.values[:, ni:(ni+step)], beta[ni:(ni+step)]))
                ni += step

        Lambda = bias_term * total_rate

        # Define Poisson likelihood
        print('creating y')
        # if fov_terms > 0:
        pm.Poisson("y", mu=area*duration*Lambda, observed=count.astype(int))
        # else:
        #    alpha = pm.Exponential('alpha', 0.01)
        #    y = pm.NegativeBinomial("y", mu=area*duration*Lambda, alpha=alpha, observed=count.astype(int))

        # sample
        from pymc.sampling_jax import sample_numpyro_nuts
        # MCMC
        draws = 10000
        chains = 4
        tune = 5000
        idata = sample_numpyro_nuts(draws, tune=tune, chains=chains,
                                    idata_kwargs={'log_likelihood': False})
        # don't include y in prior as extreme values of the coefficients make the rate of the
        # Poisson explode
        # idata.extend(pm.sample_prior_predictive(10000))
        n_samples = draws*chains
        # only use 1000 samples to get posterior predictive.
        idata_thin = idata.sel(draw=slice(None, None, int(n_samples/1000)))
        pp = None
        try:
            pm.sample_posterior_predictive(idata_thin, extend_inferencedata=True)
            pp = idata_thin.posterior_predictive
        except ValueError as e:
            print(e)
        try:
            MAP = pm.find_MAP(method='L-BFGS-B', maxeval=100000, tol=1e-5)
        except ValueError:
            print("something in the MAP didn't work")
            MAP = None

    # return results
    return {'idata': idata, 'map': MAP, 'max_area': max_area,
            'pp': pp, 'predictors': X.columns.values, 'adjust': {'mean': mean, 'scale': scale}}


# function to get get polygons which partition the FOV
def get_polygons(fov, lon, transform=True, plot=True, n_points=1000):

    # specify different projections
    aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs
    # geo = pyproj.Proj(proj='geos', h=35785831, lon_0=lon).srs
    laea = pyproj.Proj(proj='laea', lon_0=lon).srs
    cea = pyproj.Proj(proj='cea', ellps='WGS84', datum='WGS84', lon_0=lon).srs

    # FOV is given in aeqd projection, convert it to laea
    gdf = GeoDataFrame(geometry=[fov], crs=aeqd)
    fov = gdf.to_crs(laea).geometry[0]

    # partition the fov
    polygons = random_partition(fov, n_points=n_points, iterations=5)
    gdf = GeoDataFrame(geometry=polygons, crs=laea)
    # optionally plot it
    if plot:
        gdf.plot()
        plt.show()
    # optionally transform the polygons to cea
    if transform:
        gdf = gdf.to_crs(cea)
    # optionally plot the transformed data
    if plot:
        gdf.plot()
        plt.show()
    # return the polygons
    polygons = gdf.geometry
    return polygons


# discretize the dataframe of bolides in space and time
def discretize(bdf, fov=None, lon=None, showers=[], return_poly=False, n_points=1000):
    print('making polygons')

    # get polygons in the cylyndrical equal area projection, and compute their areas
    cea = pyproj.Proj(proj='cea', ellps='WGS84', datum='WGS84', lon_0=lon).srs
    polygons = get_polygons(fov, lon, transform=True, n_points=n_points, plot=False)
    areas = get_areas(polygons, cea)
    print(f'{len(polygons)} polygons made')

    # count the number of bolides in each polygon
    print('counting points')
    bdf_cea = bdf.to_crs(cea)
    counts = []
    # if we are including showers, split the bolides apart according to shower
    if len(showers) > 0:
        sdf = pd.read_csv('data/showers.csv')
        sdf.__class__ = ShowerDataFrame
        # sdf = ShowerDataFrame(source='csv', file='data/showers.csv')
        sdf = sdf[sdf.References.str.contains('2016, Icarus, 266')]
        bdfs = [bdf_cea.filter_shower(shower=showers, exclude=True, padding=5, sdf=sdf)]
        bdfs += [bdf_cea.filter_shower(shower=s, padding=10, sdf=sdf) for s in showers]
        is_shower = np.zeros((len(polygons)*(len(showers)+1), len(showers)))
        durations = [(365.25-(10*len(showers)))/365.25]*len(polygons)
        for i, bdf in enumerate(bdfs):
            counts += [sum([p.within(poly) for p in bdf.geometry]) for poly in tqdm(polygons)]
            if i >= 1:
                is_shower[:, i-1] = [0]*i*len(polygons) + [1]*len(polygons) + [0]*(len(showers)-i)*len(polygons)
                durations += [10/365.25]*len(polygons)
        shower_data_dict = dict(zip(showers, list(is_shower.T)))
    # otherwise, just count without showers.
    else:
        durations = [1] * len(polygons)
        counts = [sum([p.within(poly) for p in bdf_cea.geometry]) for poly in tqdm(polygons)]
        shower_data_dict = {}

    durations = np.array(durations).astype(float)
    counts = np.array(counts)
    # plt.hist(counts[:len(polygons)])
    # plt.show()
    # plt.hist(counts[len(polygons):(2*len(polygons))])
    # plt.show()

    # print(shower_data_dict)

    areas = areas*(len(showers)+1)  # repeat areas
    polygons = list(polygons)*(len(showers)+1)  # repeat polygons

    # compute proportion in stereo region
    print('computing proportion in stereo region')
    stereo_region = get_boundary(['goes-w', 'goes-e'], crs=cea, collection=False, intersection=True)
    stereos = np.array([poly.intersection(stereo_region).area/poly.area for poly in polygons])

    # adjust durations and stereo for GOES-17
    if lon == GOES_W_LON:
        print('adjusting g17 durations')
        fov_i, fov_ni = get_boundary(['goes-w-i', 'goes-w-ni'], crs=cea)

        # XOR representing that the point is in one but not the other
        half_observed = fov_i.symmetric_difference(fov_ni)
        affected_prop = np.array([poly.intersection(half_observed).area/poly.area for poly in polygons])

        # halve the duration for any polygons in one FOV but not the other
        durations *= 1-(affected_prop)/2

    # test gdf for plotting
    # gdf_poly = GeoDataFrame(data={'counts': counts, 'areas': areas, 'durations': durations},
    #                         geometry=polygons, crs=cea)
    # gdf_poly['density'] = gdf_poly.counts/(gdf_poly.areas*gdf_poly.durations)
    # gdf_poly.plot('density')
    # plt.show()

    # create dataframe from the computed data
    data_dict = {'count': counts, 'area': areas, 'duration': durations, 'stereo': stereos}
    data_dict = dict(data_dict, **shower_data_dict)
    gdf = GeoDataFrame(data_dict, geometry=polygons, crs=cea)

    # points = gdf.to_crs('epsg:4326').geometry

    return gdf


# define the full model
def full_model(g16=None, g17=None, separate=False, n_points=1000,
               f_lat='', f_fov='', biases=[], showers=[],
               shower_data=None, reg=True):
    # get the fields of view for GOES E and GOES W
    goes_e_fov = get_boundary('goes-e')
    goes_w_fov = get_boundary('goes-w')

    # specify GOES E and GOES W nadir
    from bolides.constants import GOES_E_LON, GOES_W_LON
    goes_e = (0, GOES_E_LON)
    goes_w = (0, GOES_W_LON)

    # collect data
    datas = []
    sats = ['g16', 'g17']
    # if 'data.pkl' not in os.listdir('models'):
    for num, bdf in enumerate([g16, g17]):
        # generate the data for each satellite separately
        fov = [goes_e_fov, goes_w_fov][num]
        fov_center = [goes_e, goes_w][num]
        gdf = discretize(bdf, fov=fov, lon=fov_center[1], n_points=n_points, showers=showers)
        data = get_data(gdf, fov_center=fov_center, shower_data=shower_data)
        data['sat'] = sats[num]
        datas.append(data)
    #    with open('models/data.pkl','wb') as f:
    #        pickle.dump(datas, f)
    # with open('models/data.pkl','rb') as f:
    #    datas = pickle.load(f)
    shower_known = (shower_data is not None)
    if not separate:
        # if not fitting separately, put the data for the two satellites together and fit
        datas = [d.to_crs(datas[0].crs) for d in datas]
        data = GeoDataFrame(pd.concat(datas, ignore_index=True), crs=datas[0].crs)
        result = fit(data, f_lat, f_fov, biases, showers, shower_known, reg)
        return dict(results=[result], data=data, f_lat=f_lat, f_fov=f_fov, biases=biases)
    # otherwise, do the fits separately
    else:
        results = []
        for data in datas:
            result = fit(data, f_lat, f_fov, biases, showers, shower_known)
            results.append(result)
        datas = [d.to_crs(datas[0].crs) for d in datas]
        data = GeoDataFrame(pd.concat(datas, ignore_index=True), crs=datas[0].crs)
        return dict(results=results, data=data, f_lat=f_lat, f_fov=f_fov, biases=biases)
