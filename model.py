import math
from math import sqrt, pi, acos, cos, radians, degrees, asin, sin
import datetime
import random
from shapely.geometry import Point, LineString, Polygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import tikzplotlib
from geopandas import GeoDataFrame
from haversine import haversine_vector
import pandas as pd
from global_land_mask import globe
import pymc as pm
import arviz as az
from tqdm import tqdm
import pyproj

from shapely import affinity
from shapely.ops import polygonize
from scipy.spatial import Voronoi

from bolides.fov_utils import get_boundary
from bolides.constants import GOES_E_LON, GOES_W_LON, GLM_STEREO_MIDPOINT
from bolides import ShowerDataFrame

import aesara.tensor as tt


def points_within(number, polygon, bbox=False):
    """Get points within a given polygon

    Parameters
    ----------
    number : int
        Number of points
    polygon : Shapely Polygon
        Polygon to get points within
    bbox : bool
        Whether to keep points falling outside the polygon

    Returns
    -------
    list of Shapely Point
        Points within the polygon
    """
    points = []
    if bbox:
        polygon = polygon.buffer(sqrt(polygon.area/pi)/10)
    minx, miny, maxx, maxy = polygon.bounds
    while len(points) < number:
        pnt = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(pnt) or bbox:
            points.append(pnt)
    return points


def idx_from_latlon(lat, lon):
    lon_idx = math.floor((lon+180)*2)
    lat_idx = math.floor((lat+90)*2)
    return lat_idx, lon_idx


def get_flash_density(flash_data, lat, lon):
    lat_idx, lon_idx = idx_from_latlon(lat, lon)
    try:
        output = flash_data['HRFC_COM_FR'][lat_idx][lon_idx]
        return output
    except IndexError:
        print(lat_idx, lon_idx, lat, lon)


def get_pixel(mapping, lat, lon):
    lat_diff = np.abs(mapping.meanLatitude - lat)
    lon_diff = np.abs(mapping.meanLongitude - lon)
    diff = lat_diff + lon_diff
    idx = np.argmin(diff)
    if min(diff) > 2:
        raise ValueError('Difference from pixel center is greater than 2. Is this the right mapping?')
    return mapping.x.iloc[idx], mapping.y.iloc[idx]


def get_pitch(mapping, lat, lon):
    x_pitches = np.array([30, 28, 26, 24])
    x_thresh = np.array([280, 344, 408])
    y_pitches = np.array([30, 28, 26, 24, 22, 20])
    y_thresh = np.array([280, 344, 408, 472, 536])

    x, y = get_pixel(mapping, lat, lon)
    quad_x = abs(x-650)
    quad_y = abs(y-686)

    x_pitch = x_pitches[np.argmin(quad_x > x_thresh)]
    if all(quad_x > x_thresh):
        x_pitch = 24
    y_pitch = y_pitches[np.argmin(quad_y > y_thresh)]
    if all(quad_y > y_thresh):
        y_pitch = 20

    return float(x_pitch*y_pitch)

def angle_from_nadir(lat, lon, central_lon):
    lon -= central_lon
    return degrees(acos(cos(radians(lat))*cos(radians(lon))))

def distance_to_angle(dist):
    r = 6357
    goes_height = 35786 
    alpha = dist / r
    point_dist = sqrt((goes_height+r)**2 + r**2 - 2*(goes_height+r)*r*cos(alpha))
    theta = asin(r*sin(alpha)/point_dist)
    return degrees(theta)

def get_data(points, counts, areas, fov_center=None, durations=None, shower_data_dict=None, mapping=None):
    lons = np.array([p.x for p in points])
    lats = np.array([p.y for p in points])
    coords = list(zip(lats, lons))
    if fov_center is not None:
        fov_dists = np.array(haversine_vector([fov_center]*len(coords), coords))
        fov_dists /= 10000
        # fov_dists = np.array([get_view_angle(lat, lon, fov_center[1]) for lat, lon in coords])
        # fov_dists /= 10 # normalize
    if mapping is not None:
        print('getting pixel pitches')
        pitches = np.array([get_pitch(mapping, p.y, p.x) for p in tqdm(points)])
        pitches /= 900

    from netCDF4 import Dataset
    flash_data = Dataset('data/LISOTD_HRFC_V2.3.2015.nc')
    flash_dens = np.array([get_flash_density(flash_data, p.y, p.x) for p in points])
    flash_dens = np.nan_to_num(flash_dens, copy=True, nan=0.0)

    land = np.array([globe.is_land(lat, lon) for lat, lon in zip(lats, lons)])
    land = land.astype(int)

    # normalize so that 1 is the maximum value
    flash_dens /= max(flash_dens)  # todo: should this scaling really be different for the two satellites?


    # TESTING, DO NOT USE
    # lats = lons - fov_center[1]
    # TODO: DELETE THE ABOVE LINE

    lats /= 90

    if fov_center is not None:
        data_dict = {'lat': lats, 'fov_dist': fov_dists, 'flash_dens': flash_dens,
                     'land': land, 'area': areas, 'duration': durations}
        data_dict = dict(data_dict, **shower_data_dict)
        data = pd.DataFrame(data_dict)
    else:
        data = pd.DataFrame({'lat': np.abs(lats), 'lat2': np.abs(lats)**2, 'lat4': lats**4,
                             'flash_dens': flash_dens, 'land': land, 'area': areas})

    data['count'] = counts
    return data


def fit(data, **kwargs):

    lat = data['lat']
    fov = data['fov_dist']
    flash = data['flash_dens']
    area = data['area'].values
    area_normalizer = max(area)
    area = area/area_normalizer
    count = data['count'].values
    duration = data['duration'].values
    showers = [col for col in data.columns if col not in ['lat', 'fov_dist', 'flash_dens', 'area', 'count', 'land', 'duration']]
    plt.hist(count)
    plt.show()

    with pm.Model() as mdl_fish:

        # define priors, weakly informative Normal
        intercept = pm.Normal("intercept", mu=0, sigma=200)
        l1 = pm.Normal("lat1", mu=0, sigma=200)
        # labs = pm.Normal("latabs", mu=0, sigma=200)
        l2 = pm.Normal("lat2", mu=0, sigma=200)
        l3 = pm.Normal("lat3", mu=0, sigma=200)
        f1 = pm.Normal("fov_dist", mu=0, sigma=200)
        f2 = pm.Normal("fov_dist2", mu=0, sigma=200)
        f3 = pm.Normal("fov_dist3", mu=0, sigma=200)
        # fl = pm.Normal("flash_dens", mu=0, sigma=200)
        # p1 = pm.Normal("pitch", mu=0, sigma=200)
        # p2 = pm.Normal("pitch2", mu=0, sigma=200)
        # b7 = pm.Normal("land", mu=0, sigma=200)

        # define linear model and exp link function
        theta = intercept + l1*lat + l2*lat**2 + l3*np.abs(lat**3) 
        if 'fov_dist' in data.columns:
            fov_theta = f1*fov + f2*fov**2 + f3*fov**3
        else:
            theta = intercept + l1*lat + l2*lat**2 + l3*lat**3 + fl*flash

        thetas = []
        indicators = []
        if len(showers) > 0:
            for s in showers:
                print(s)
                intercept = pm.Normal(s+"intercept", mu=0, sigma=200)
                l1 = pm.Normal(s+"lat1", mu=0, sigma=200)
                l2 = pm.Normal(s+"lat2", mu=0, sigma=200)
                l3 = pm.Normal(s+"lat3", mu=0, sigma=200)
                indicators.append(np.array(data[s]))
                print(data[s])
                newtheta = intercept + l1*lat + l2*lat**2 + l3*np.abs(lat**3)
                thetas.append(newtheta)


        # Define Poisson likelihood
        print('creating y')
        showersum = indicators[0]*np.exp(thetas[0])
        for i in range(1, len(thetas)):
            showersum += indicators[i]*np.exp(thetas[i])
        y = pm.Poisson("y", mu=area*duration*np.exp(fov_theta)*(np.exp(theta)+showersum), observed=count)

    with mdl_fish:
        # result = pm.find_MAP()
        result_pos = pm.sample(1000, tune=200, cores=4, return_inferencedata=True, target_accept=0.95)
        result_map = pm.find_MAP()

    return {'posterior':result_pos, 'map': result_map, 'area_normalizer': area_normalizer}

def fit_nonparam(data, **kwargs):

    lat = data['lat']
    fov = data['fov_dist']
    flash = data['flash_dens']
    area = data['area'].values
    count = data['count'].values
    duration = data['duration'].values
    showers = [col for col in data.columns if col not in ['lat', 'fov_dist', 'flash_dens', 'area', 'count', 'land', 'duration']]
    plt.hist(count)
    plt.show()

    n_u = 10
    X = np.array(lat)[:,None]
    Xu = np.linspace(0, 2/3, n_u)[:,None]

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

        #l = pm.Gamma("l", alpha=2, beta=1)
        #eta = pm.HalfNormal("eta", sigma=1)
        #cov = eta**2 * pm.gp.cov.ExpQuad(1, l)
        #Kuu = cov(Xu)
        #L = pm.gp.util.cholesky(pm.gp.util.stabilize(Kuu))
        #v = pm.Normal("u_rotated_", mu=0.0, sigma=1.0, shape=n_u)
        #u = pm.Deterministic("u", tt.dot(L, v))
        #Kfu = cov(X, Xu)
        #Kuiu = tt.slinalg.solve_upper_triangular(L.T, tt.slinalg.solve_lower_triangular(L, u))
        #f = pm.Deterministic("f", tt.dot(Kfu, Kuiu))

        fov_factor = pm.Deterministic('b', np.exp(f_fov))
        lat_factor = pm.Deterministic('lambda', np.exp(f_lat))

        print('creating y')
        y = pm.Poisson("y", mu=area*duration*fov_factor*lat_factor, observed=count) #TODO: re-add fov_factor

    with mdl_fish:
        # result = pm.find_MAP()
        import pymc.sampling_jax
        result_pos = pm.sampling_jax.sample_numpyro_nuts(100, tune=100, chains=4)
        #result_pos = pm.sample(100, tune=100, cores=1, chains=1, return_inferencedata=True)
        result_map = pm.find_MAP()
    with mdl_fish:
        preds = pm.sample_posterior_predictive(result_pos, var_names=['f_fov','f_lat'])

    return {'posterior': result_pos, 'map': result_map, 'preds': preds, 'lat': lat, 'fov': fov}

def partial_fit(data, fov_posterior, **kwargs):

    lat = data['lat']
    fov = data['fov_dist']
    flash = data['flash_dens']
    area = data['area'].values
    count = data['count'].values

    results = []
    pos = fov_posterior.posterior
    for i in range(10):
        print(i)
        chain = random.randint(0, len(pos.chain)-1)
        draw = random.randint(0, len(pos.draw)-1)

        f1 = pos.fov_dist[chain][draw].data
        f2 = pos.fov_dist2[chain][draw].data
        f3 = pos.fov_dist3[chain][draw].data

        with pm.Model() as mdl_fish:

            # define priors, weakly informative Normal
            intercept = pm.Normal("intercept", mu=0, sigma=200)
            l1 = pm.Normal("lat1", mu=0, sigma=200)
            # labs = pm.Normal("latabs", mu=0, sigma=200)
            l2 = pm.Normal("lat2", mu=0, sigma=200)
            l3 = pm.Normal("lat3", mu=0, sigma=200)

            # define linear model and exp link function
            if 'fov_dist' in data.columns:
                theta = intercept + l1*lat + l2*lat**2 + l3*np.abs(lat**3) + f1*fov + f2*fov**2 + f3*fov**3 # + fl*flash
            else:
                theta = intercept + l1*lat + l2*lat**2 + l3*lat**3 + fl*flash

            # Define Poisson likelihood
            y = pm.Poisson("y", mu=area*np.exp(theta), observed=count)

        with mdl_fish:
            # result = pm.find_MAP()
            result = pm.sample(1000, tune=200, cores=4, return_inferencedata=True, target_accept=0.8)
        if i == 0:
            results = result
        else:
            results = az.concat(results, result, dim='chain')

    return results


def get_circles_and_counts(bdf, fov):
    print('getting circles')

    import pyproj
    # laea = pyproj.Proj(proj='laea', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs
    from bolides.constants import GLM_STEREO_MIDPOINT
    eck4 = pyproj.Proj(proj='eck4', ellps='WGS84', datum='WGS84', lat_0=0, lon_0=GLM_STEREO_MIDPOINT).srs

    aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs

    gdf = GeoDataFrame(geometry=[fov], crs=aeqd)
    fov = gdf.to_crs(eck4).geometry[0]
    circles = [p.buffer(sqrt(fov.area/(1000*pi))) for p in points_within(10000, fov)]

    print('keeping good circles')
    good_circles = []
    for circle in circles:
        if circle.intersection(fov).area >= circle.area * 0.999:
            good_circles.append(circle)

    print('counting points in circles')

    bdf_eck4 = bdf.to_crs(eck4)
    point_counts = np.array([sum([p.within(circle) for p in bdf_eck4.geometry]) for circle in good_circles])

    gdf = GeoDataFrame(data={'counts': point_counts}, geometry=good_circles, crs=eck4)
    centroids = [list(c.centroid.xy) for c in good_circles]
    xs = [c[0][0] for c in centroids]
    ys = [c[1][0] for c in centroids]
    points = [Point(x, y) for x, y in zip(xs, ys)]
    gdf = GeoDataFrame(geometry=points, crs=eck4)
    points = gdf.to_crs('epsg:4326').geometry

    return points, point_counts


def get_polygons(fov, lon, transform=True, plot=True, n_points=1000):
    aeqd = pyproj.Proj(proj='aeqd', ellps='WGS84', datum='WGS84', lat_0=90, lon_0=0).srs
    geo = pyproj.Proj(proj='geos', h=35785831, lon_0=lon).srs
    eck4 = pyproj.Proj(proj='eck4', ellps='WGS84', datum='WGS84', lat_0=0, lon_0=lon).srs

    gdf = GeoDataFrame(geometry=[fov], crs=aeqd)
    fov = gdf.to_crs(geo).geometry[0]

    # Voronoi doesn't work properly with points below (0,0) so set lowest point to (0,0)
    x_shift = fov.bounds[0]
    y_shift = fov.bounds[1]

    shape = fov
    #shape = affinity.translate(fov, -x_shift, -y_shift)
    points = points_within(n_points, shape, bbox=True) #TODO: change back to 2000
    #points += [shape.boundary.interpolate(dist, normalized=True) for dist in np.linspace(0, 1, 1000)]
    min_x = min([p.x for p in points])
    min_y = min([p.y for p in points])
    coords = list(zip([p.x-min_x for p in points], [p.y-min_y for p in points]))
    shape = affinity.translate(fov, -min_x, -min_y)
    for i in range(5):
        vor = Voronoi(coords)
        lines = [LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
        polygons = list(polygonize(lines))
        polygons = [poly.intersection(shape) for poly in polygons]
        good_polygons = []
        for poly in polygons:
            if poly.area >= shape.area/len(polygons)/20:
                good_polygons.append(poly)
        polygons = good_polygons
        points = [poly.centroid for poly in polygons]
        #coords = list(zip([p.x for p in points], [p.y for p in points]))

    polygons = [affinity.translate(poly, min_x, min_y) for poly in polygons]
    gdf = GeoDataFrame(geometry=polygons, crs=geo)
    if plot:
        gdf.plot()
        plt.show()
    if transform:
        gdf = gdf.to_crs(eck4)
    polygons = gdf.geometry
    return polygons


def get_polygons_globe():
    eck4 = pyproj.Proj(proj='eck4', ellps='WGS84', datum='WGS84', lat_0=0, lon_0=0).srs

    fov = Polygon([[-180, 90], [180, 90], [180, -90], [-180, -90]])

    # Voronoi doesn't work properly with points below (0,0) so set lowest point to (0,0)
    x_shift = fov.bounds[0]
    y_shift = fov.bounds[1]

    shape = affinity.translate(fov, -x_shift, -y_shift)
    points = points_within(1000, shape)
    points += [shape.boundary.interpolate(dist, normalized=True) for dist in np.linspace(0, 1, 1000)]
    coords = list(zip([p.x for p in points], [p.y for p in points]))
    for i in range(5):
        vor = Voronoi(coords)
        lines = [LineString(vor.vertices[line]) for line in vor.ridge_vertices if -1 not in line]
        polygons = list(polygonize(lines))
        polygons = [poly.intersection(shape) for poly in polygons]
        points = [poly.centroid for poly in polygons]
        coords = list(zip([p.x for p in points], [p.y for p in points]))

    polygons = [affinity.translate(poly, x_shift, y_shift) for poly in polygons]
    gdf = GeoDataFrame(geometry=polygons, crs='epsg:4326')
    polygons = gdf.to_crs(eck4).geometry
    return polygons


def get_polygons_and_counts(bdf, fov=None, lon=None, showers=[], return_poly=False, n_points=1000):
    print('making polygons')
    if fov is not None and lon is not None:
        from bolides.constants import GLM_STEREO_MIDPOINT

        eck4 = pyproj.Proj(proj='eck4', ellps='WGS84', datum='WGS84', lat_0=0, lon_0=lon).srs

        polygons = get_polygons(fov, lon, n_points=n_points)
        areas = [poly.area for poly in polygons]
    else:
        eck4 = pyproj.Proj(proj='eck4', ellps='WGS84', datum='WGS84', lat_0=0, lon_0=0).srs
        polygons = get_polygons_globe()
        areas = [poly.area for poly in polygons]
    print(f'{len(polygons)} polygons made')

    print('counting points')
    bdf_eck4 = bdf.to_crs(eck4)
    counts = []
    if len(showers)>0:
        sdf = ShowerDataFrame()
        sdf = sdf[sdf.References.str.contains('2016, Icarus, 266')]
        bdfs = [bdf_eck4.filter_shower(shower=showers, exclude=True, padding=10)]
        bdfs += [bdf_eck4.filter_shower(shower=s, padding=10) for s in showers]
        is_shower = np.zeros((len(polygons)*(len(showers)+1), len(showers)))
        durations = [(365.25-(10*len(showers)))/365.25]*len(polygons)
        for i, bdf in enumerate(bdfs):
            counts += [sum([p.within(poly) for p in bdf.geometry]) for poly in tqdm(polygons)]
            if i >= 1:
                is_shower[:,i-1] = [0]*i*len(polygons) + [1]*len(polygons) + [0]*(len(showers)-i)*len(polygons)
                durations += [10/365.25]*len(polygons)
        shower_data_dict = dict(zip(showers, list(is_shower.T)))
    else:
        durations = [1] * len(polygons)
        counts = [sum([p.within(poly) for p in bdf_eck4.geometry]) for poly in tqdm(polygons)]
        shower_data_dict = {}

    durations = np.array(durations).astype(float)
    counts = np.array(counts)
    plt.hist(counts[:len(polygons)])
    plt.show()
    plt.hist(counts[len(polygons):(2*len(polygons))])
    plt.show()

    print(shower_data_dict)

    areas = areas*(len(showers)+1) # repeat areas
    polygons = list(polygons)*(len(showers)+1) # repeat polygons

    centroids = [list(c.centroid.xy) for c in polygons]
    xs = [c[0][0] for c in centroids]
    ys = [c[1][0] for c in centroids]
    points = [Point(x, y) for x, y in zip(xs, ys)]

    # adjust durations for GOES-17
    if lon == GOES_W_LON:
        print('adjusting g17 durations')
        fov_i, fov_ni = get_boundary(['goes-w-i','goes-w-ni'], crs=eck4)
        within = [(p.within(fov_i) ^ p.within(fov_ni)) for p in points] #XOR representing that the point is in one but not the other
        # halve the duration for any polygons in one FOV but not the other
        durations[within] *= 0.5

    # test gdf for plotting
    gdf_poly = GeoDataFrame(data={'counts': counts, 'areas': areas, 'durations': durations}, geometry=polygons, crs=eck4)
    gdf_poly['density'] = gdf_poly.counts/(gdf_poly.areas*gdf_poly.durations)
    gdf_poly.plot('density')
    plt.show()

    gdf = GeoDataFrame(geometry=points, crs=eck4)
    points = gdf.to_crs('epsg:4326').geometry


    if not return_poly:
        return points, counts, areas, durations, shower_data_dict
    else:
        return gdf_poly, points


def full_model(g16, g17, separate=False, nonparam=False, n_points=1000, showers=[]):
    goes_e_fov = get_boundary('goes-e')
    goes_w_fov = get_boundary('goes-w')
    from bolides.constants import GOES_E_LON, GOES_W_LON
    goes_e = (0, GOES_E_LON)
    goes_w = (0, GOES_W_LON)

    fit_func = fit_nonparam if nonparam else fit

    # TABLE_PATH = '/home/aozerov/projects/seti/nav-tables'
    # pixels_g16 = pd.read_csv(f'{TABLE_PATH}/G16_nav.csv')
    # pixels_g17 = pd.read_csv(f'{TABLE_PATH}/G17_nav_approx.csv')

    datas = []
    for num, bdf in enumerate([g16, g17]):
        fov = [goes_e_fov, goes_w_fov][num]
        fov_center = [goes_e, goes_w][num]
        # mapping = [pixels_g16, pixels_g17][num]
        points, point_counts, areas, durations, shower_data_dict = get_polygons_and_counts(bdf, fov=fov, lon=fov_center[1], n_points=n_points, showers=showers)
        datas.append(get_data(points, point_counts, areas, fov_center, durations, shower_data_dict))  # , mapping))
    if not separate:
        data = pd.concat(datas, ignore_index=True)
        result = fit_func(data)
        return result
    else:
        results = []
        for data in datas:
            result = fit_func(data)
            results.append(result)
        return results

def partial_model(g16, g17, fov_posterior, separate=False):
    goes_e_fov = get_boundary('goes-e')
    goes_w_fov = get_boundary('goes-w')
    goes_e = (0, GOES_E_LON)
    goes_w = (0, GOES_W_LON)

    # TABLE_PATH = '/home/aozerov/projects/seti/nav-tables'
    # pixels_g16 = pd.read_csv(f'{TABLE_PATH}/G16_nav.csv')
    # pixels_g17 = pd.read_csv(f'{TABLE_PATH}/G17_nav_approx.csv')

    datas = []
    for num, bdf in enumerate([g16, g17]):
        fov = [goes_e_fov, goes_w_fov][num]
        fov_center = [goes_e, goes_w][num]
        # mapping = [pixels_g16, pixels_g17][num]
        points, point_counts, areas = get_polygons_and_counts(bdf, fov, fov_center[1])
        datas.append(get_data(points, point_counts, areas, fov_center))  # , mapping))
    if not separate:
        data = pd.concat(datas, ignore_index=True)
        result = partial_fit(data, fov_posterior)
        return result
    else:
        results = []
        for data in datas:
            result = partial_fit(data, fov_posterior)
            results.append(result)
        return results

def full_model_usg(bdf):

    points, point_counts, areas = get_polygons_and_counts(bdf)
    data = get_data(points, point_counts, areas)
    print(len(data))
    result = fit(data)
    return result

def plot_polygons(gdf, filename):
    gdf.plot()
    fig = plt.gcf()
    fig.axes[0].xaxis.set_visible(False)
    fig.axes[0].yaxis.set_visible(False)
    #print(dir(fig))
    plt.savefig(filename+'.svg', bbox_inches='tight')
    plt.show()
