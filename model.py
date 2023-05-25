import numpy as np
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame
from haversine import haversine_vector
import pandas as pd
from global_land_mask import globe
import pymc as pm
from tqdm import tqdm
import pyproj
import cartopy.crs as ccrs

from shapely.geometry import Point

from bolides.fov_utils import get_boundary
from bolides import ShowerDataFrame
from bolides.constants import GOES_W_LON

from geo_utils import get_pitch, get_flash_density, get_areas
from partition import random_partition


# function to add lons, lats, distances from nadir, and other
# variables to the dataframe of polygons and counts.
def get_data(gdf, fov_center=None, mapping=None, ecliptic=False):

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
        fov_dists = np.array(haversine_vector([fov_center]*len(coords), coords))
        fov_dists /= 10000

    # get the pixel pitches of the pixels on the CCD corresponding to the centroids
    if mapping is not None:
        print('getting pixel pitches')
        pitches = np.array([get_pitch(mapping, p.y, p.x) for p in tqdm(points)])
        pitches /= 900


    from netCDF4 import Dataset
    flash_data = Dataset('data/LISOTD_HRFC_V2.3.2015.nc')
    flash_dens = np.array([get_flash_density(flash_data, lat, lon) for lat, lon in zip(lats, lons)])
    #flash_dens = np.nan_to_num(flash_dens, copy=True, nan=0.0)

    # calculate whether a centroid is over land or not
    land = np.array([globe.is_land(lat, lon) for lat, lon in zip(lats, lons)])
    land = land.astype(int)

    # normalize so that 1 is the maximum value
    #flash_dens /= max(flash_dens)  # todo: should this scaling really be different for the two satellites?

    if ecliptic:
        #TODO: this won't work. Different points in same polygon hit at different ecliptic latitudes
        print('transforming coords to ecliptic')
        from astropy.coordinates import ICRS, SkyCoord
        from astropy.time import Time
        import astropy.units as u
        eclipticlats = []
        rows = zip(lats, lons, gdf['datetime'])
        for lat, lon, time in rows:
            c = SkyCoord(ra=lon*u.degree, dec=lat*u.degree, obstime=Time(time), frame='itrs')
            icrs = c.transform_to(ICRS)
            eclipticlats.append(icrs.barycentrictrueecliptic.lat.value)
        lats = np.array(eclipticlats)

    # normalize to [0,1]
    lats /= 90

    # add new columns to the dataframe
    gdf['lat'] = lats
    gdf['lon'] = lons
    gdf['fov_dist'] = fov_dists
    gdf['flash_dens'] = flash_dens
    gdf['land'] = land

    return gdf

# given dataset, fit the model
def fit(data, f_lat, f_fov, biases, **kwargs):
    # extract variables from the data
    lat = np.array(data['lat'])
    fov = np.array(data['fov_dist'])
    area = np.array(data['area'])
    max_area = max(area)
    area = area/max_area
    count = np.array(data['count'])
    duration = np.array(data['duration'])
    non_shower_cols = ['lat', 'lon', 'fov_dist', 'flash_dens', 'area', 'count', 'land', 'duration', 'geometry', 'sat']
    showers = [col for col in data.columns if col not in non_shower_cols]

    # define model
    with pm.Model() as model:

        var_names = []

        fov_term = 0
        if len(f_fov)>0:
            for term in f_fov.split('+'):
                power = int(term.split('^')[1])
                fovpower = fov**power
                fov_term += pm.Normal(f'fov{power}', 0, 1/np.var(fovpower))*fovpower
                #fov_term += pm.Cauchy(f'fov{power}', alpha=0, beta=0.5)*(fov**power)
                var_names.append(f'fov{power}')
                print(f'added fov^{power} term')

        lat_term = 0
        if len(f_lat)>0:
            for term in f_lat.split('+'):
                power = int(term.split('^')[1])
                latpower = lat**power
                lat_term += pm.Normal(f'lat{power}', 0, 1/np.var(latpower))*latpower
                #lat_term += pm.Cauchy(f'lat{power}', alpha=0, beta=0.5)*(lat**power)
                var_names.append(f'lat{power}')
                print(f'added lat^{power} term')

        #intercept = pm.Normal("intercept", 0, 10)
        intercept = pm.Cauchy("intercept", alpha=0, beta=0.5)

        bias_term = 0
        for bias in biases:
            bias_term += pm.Normal(bias, 0, 10)*np.array(data[bias]).astype(float)
            #bias_term += pm.Cauchy(bias, alpha=0, beta=0.5)*np.array(data[bias]).astype(float)
            var_names.append(bias)
            print(f'added {bias} term')

        # thetas = []
        # indicators = []
        # if len(showers) > 0:
        #     for s in showers:
        #         intercept = pm.Normal(s+"intercept", mu=0, sigma=1e6)
        #         l1 = pm.Normal(s+"lat1", mu=0, sigma=1e6)
        #         l2 = pm.Normal(s+"lat2", mu=0, sigma=1e6)
        #         l3 = pm.Normal(s+"lat3", mu=0, sigma=1e6)
        #         indicators.append(np.array(data[s]))
        #         newtheta = intercept + l1*lat + l2*lat**2 + l3*np.abs(lat**3)
        #         thetas.append(newtheta)

        # Define Poisson likelihood
        print('creating y')
        # if including showers, add up the shower terms
        # if len(showers) > 0:
        #     showersum = indicators[0]*np.exp(thetas[0])
        #     for i in range(1, len(thetas)):
        #         print(indicators[i])
        #         showersum += indicators[i]*np.exp(thetas[i])
        # else:
        #     showersum = 0
        # specify that y is generated by Poisson
        # y = pm.Poisson("y", mu=area*duration*np.exp(fov_theta)*(np.exp(theta)+showersum), observed=count)
        y = pm.Poisson("y", mu=area*duration*np.exp(intercept+fov_term+lat_term+bias_term), observed=count.astype(int))

        # sample
        import pymc.sampling_jax
        # MCMC
        idata = pm.sampling_jax.sample_numpyro_nuts(5000, tune=2000, chains=2)
        # don't include y in prior as extreme values of the coefficients make the rate of the
        # Poisson explode
        idata.extend(pm.sample_prior_predictive(10000, var_names=var_names))
        pm.sample_posterior_predictive(idata, extend_inferencedata=True)
        #result_pos = pm.sample(100, tune=200, cores=4, return_inferencedata=True, target_accept=0.95)
        # maximum a posteriori
        MAP = pm.find_MAP(method='L-BFGS-B')

    # return results
    return {'idata': idata, 'map': MAP, 'max_area': max_area}


# nonparameteric model fit
def fit_nonparam(data, **kwargs):

    lat = np.array(data['lat'])
    fov = np.array(data['fov_dist'])
    area = np.array(data['area'])
    max_area = max(area)
    area = area/max_area
    count = np.array(data['count'])
    duration = np.array(data['duration'])

    with pm.Model() as mdl_fish:

        rho_fov = pm.Exponential('rho_fov', 1)
        eta_fov = pm.Exponential('eta_fov', 1)
        K_fov = eta_fov**2 * pm.gp.cov.ExpQuad(1, rho_fov)
        rho_lat = pm.Exponential('rho_lat', 1)
        eta_lat = pm.Exponential('eta_lat', 1)
        K_lat = eta_lat**2 * pm.gp.cov.ExpQuad(1, rho_lat)

        gp_fov = pm.gp.Latent(cov_func=K_fov)
        gp_lat = pm.gp.Latent(cov_func=K_lat)

        f_fov = gp_fov.prior('f_fov', np.array(fov)[:, None])
        f_lat = gp_lat.prior('f_lat', np.array(lat)[:, None])

        fov_factor = pm.Deterministic('b', np.exp(f_fov))
        lat_factor = pm.Deterministic('lambda', np.exp(f_lat))

        print('creating y')
        y = pm.Poisson("y", mu=area*duration*fov_factor*lat_factor, observed=count)

    with mdl_fish:
        import pymc.sampling_jax
        result_pos = pm.sample(100, tune=100, chains=1)
        #result_pos = pm.sampling_jax.sample_numpyro_nuts(100, tune=100, chains=1)
        result_map = pm.find_MAP()
    with mdl_fish:
        preds = pm.sample_posterior_predictive(result_pos, var_names=['f_fov', 'f_lat'])

    return {'posterior': result_pos, 'map': result_map, 'preds': preds, 'lat': lat, 'fov': fov}


# function to get get polygons which partition the FOV
def get_polygons(fov, lon, transform=True, plot=True, n_points=1000):

    # specify different projections
    aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs
    geo = pyproj.Proj(proj='geos', h=35785831, lon_0=lon).srs
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
        bdfs = [bdf_cea.filter_shower(shower=showers, exclude=True, padding=10, sdf=sdf)]
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
    #plt.hist(counts[:len(polygons)])
    #plt.show()
    #plt.hist(counts[len(polygons):(2*len(polygons))])
    #plt.show()

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
    #gdf_poly = GeoDataFrame(data={'counts': counts, 'areas': areas, 'durations': durations}, geometry=polygons, crs=cea)
    #gdf_poly['density'] = gdf_poly.counts/(gdf_poly.areas*gdf_poly.durations)
    #gdf_poly.plot('density')
    #plt.show()

    # create dataframe from the computed data
    data_dict = {'count': counts, 'area': areas, 'duration': durations, 'stereo': stereos}
    data_dict = dict(data_dict, **shower_data_dict)
    gdf = GeoDataFrame(data_dict, geometry=polygons, crs=cea)

    #points = gdf.to_crs('epsg:4326').geometry

    return gdf


# define the full model
def full_model(g16=None, g17=None, separate=False, nonparam=False, n_points=1000,
               f_lat='', f_fov='', biases=[], showers=[], ecliptic=False):
    # get the fields of view for GOES E and GOES W
    goes_e_fov = get_boundary('goes-e')
    goes_w_fov = get_boundary('goes-w')

    # specify GOES E and GOES W nadir
    from bolides.constants import GOES_E_LON, GOES_W_LON
    goes_e = (0, GOES_E_LON)
    goes_w = (0, GOES_W_LON)

    # specify the fitting function
    fit_func = fit_nonparam if nonparam else fit

    # collect data
    datas = []
    sats = ['g16', 'g17']
    for num, bdf in enumerate([g16, g17]):
        # generate the data for each satellite separately
        fov = [goes_e_fov, goes_w_fov][num]
        fov_center = [goes_e, goes_w][num]
        gdf = discretize(bdf, fov=fov, lon=fov_center[1], n_points=n_points, showers=showers)
        data = get_data(gdf, fov_center=fov_center, ecliptic=ecliptic)
        data['sat'] = sats[num]
        datas.append(data)
    if not separate:
        # if not fitting separately, put the data for the two satellites together and fit
        datas = [d.to_crs(datas[0].crs) for d in datas]
        data = GeoDataFrame(pd.concat(datas, ignore_index=True), crs=datas[0].crs)
        result = fit_func(data, f_lat, f_fov, biases)
        return dict(results=[result], data=data, f_lat=f_lat, f_fov=f_fov, biases=biases)
    # otherwise, do the fits separately
    else:
        results = []
        for data in datas:
            result = fit_func(data, f_lat, f_fov, biases)
            results.append(result)
        data = GeoDataFrame(pd.concat(datas, ignore_index=True), crs=datas[0].crs)
        return dict(results=results, data=data)
