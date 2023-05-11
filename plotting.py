import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame
import numpy as np
import random
import cartopy.crs as ccrs

from shapely.geometry import Point

from matplotlib.lines import Line2D
import matplotlib.cm as cmx
from matplotlib.colors import Normalize

from geo_utils import distance_to_angle

from bolides.constants import GLM_STEREO_MIDPOINT, GOES_E_LON
import bolides.crs as bcrs


def plot_polygons(gdf, filename=None, column='area', label='Polygon area (km$^{-2}$)', 
                  crs=ccrs.EckertIV(central_longitude=GLM_STEREO_MIDPOINT), insetcrs=bcrs.GOES_E(),
                  show=False):

    plt.style.use('default')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    crs_proj4 = crs.proj4_init
    gdf = gdf.to_crs(crs_proj4)
    fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize=(10,4))
    ax.stock_img()
    cmap = plt.get_cmap('viridis')
    if column == 'density':
        data = gdf['count']/gdf['area']
    else:
        data = gdf[column]
    cNorm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    colors = [scalarMap.to_rgba(num) for num in data]
    for num, poly in enumerate(gdf.geometry):
        ax.add_geometries([poly], crs=crs, color=colors[num], linewidth=0)

    points = [Point(lon, lat) for lon, lat in zip(gdf['lon'], gdf['lat']*90)]
    points = GeoDataFrame(geometry=points, crs='epsg:4326').to_crs(crs.proj4_init).geometry
    x = np.array([p.x for p in points])
    y = np.array([p.y for p in points])
    plt.scatter(x, y, color='red', marker='.', s=4, zorder=5, edgecolor='none')
    plt.colorbar(scalarMap, label=label)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import cartopy
    axins = inset_axes(ax, width="100%", height="100%", loc='lower left',
                       bbox_to_anchor=(0.5, -0.05, 0.75, 0.6),
                       bbox_transform=ax.transAxes,
                       axes_class=cartopy.mpl.geoaxes.GeoAxes,
                       axes_kwargs=dict(map_projection=insetcrs))
    axins.stock_img()
    for num, poly in enumerate(gdf.geometry):
        axins.add_geometries([poly], crs=crs, color=colors[num], linewidth=0)
    points = [Point(lon, lat) for lon, lat in zip(gdf['lon'], gdf['lat']*90)]
    points = GeoDataFrame(geometry=points, crs='epsg:4326').to_crs(insetcrs.proj4_init).geometry
    x = np.array([p.x for p in points])
    y = np.array([p.y for p in points])
    plt.scatter(x, y, color='red', marker='.', s=3, zorder=5, edgecolor='none')
    axins.coastlines()
    axins.gridlines(color='black',draw_labels=False, xlocs=[0,60,120,180,-60,-120])

    ax.coastlines()
    ax.gridlines(color='black',draw_labels=True)

    if filename is not None:
        plt.savefig(f'data-plots/{filename}-{column}.pdf', bbox_inches='tight')
    if show:
        plt.show()
    # gdf.plot(column)
    # plt.show()


def get_varfactor(f, var, x, shower='', m_ap=None, pos=None, chain=None, draw=None):
    assert m_ap is not None or pos is not None
    if pos is not None:
        assert chain is not None and draw is not None

    varfactor = np.zeros(len(x))
    for term in f.split('+'):
        power = int(term.split('^')[1])
        key = shower+var+str(power)

        if m_ap is not None:
            varfactor += m_ap[key]
        elif pos is not None:
            varfactor += pos[key][chain][draw].data * x**power
    return varfactor


def plot_lat_result(result, title, filename, plot_map=True, max_lat=55, normalize=True, symmetric=False, shower='', theory=None, show=False):
    plt.style.use('default')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, ax1 = plt.subplots()
    ax1.xaxis.set_tick_params(which='major', top='on')
    ax1.yaxis.set_tick_params(which='major', top='on')

    f_lat = result['f_lat']
    result = result['results'][0]
    pos = result['idata'].posterior
    m_ap = result['map']
    max_area = result['max_area']

    top_y = 0
    y_plots = np.zeros((400, 200//(int(symmetric)+1)))
    x_plot = np.linspace(-max_lat, max_lat, 200)/90
    for i in range(401):
        chain = random.randint(0, len(pos.chain)-1)
        draw = random.randint(0, len(pos.draw)-1)

        if i != 400:
            color = 'black'
            alpha = 0.2
            linewidth = 0.5

            intercept = pos.__getattr__(shower+"intercept")[chain][draw].data
            varfactor = get_varfactor(f_lat, 'lat', x_plot, shower=shower, pos=pos, chain=chain, draw=draw)
        else:
            color = 'red'
            alpha = 1
            linewidth = 1

            intercept = m_ap['intercept']
            varfactor = get_varfactor(f_lat, 'lat', x_plot, shower=shower, m_ap=m_ap)

        if normalize:
            y_plot = np.exp(varfactor)
        else:
            y_plot = np.exp(intercept + varfactor)/max_area
        y_plot = np.array(y_plot)
        if symmetric:
            midpoint = int(len(x_plot)/2)
            x_plot = x_plot[midpoint:]
            y_plot = (y_plot[midpoint:] + np.flipud(y_plot[:midpoint]))/2
        top_y = max(max(y_plot), top_y)
        if i != 400:
            y_plots[i, :] = y_plot
        if not (i==400 and not plot_map):
            ax1.plot(x_plot*90, y_plot, color=color, alpha=alpha, linewidth=linewidth)

    # plot quantiles
    top_quantile = np.quantile(y_plots, 0.90, axis=0)
    bottom_quantile = np.quantile(y_plots, 0.10, axis=0)
    median = np.quantile(y_plots, 0.5, axis=0)
    plt.plot(x_plot*90, top_quantile, color='red', linewidth=1, linestyle='--')
    plt.plot(x_plot*90, bottom_quantile, color='red', linewidth=1, linestyle='--')
    plt.plot(x_plot*90, median, color='red', linewidth=1, linestyle='-')

    plt.xlabel('Latitude (°)')
    plt.ylabel('Normalized bolide flux')
    try:
        plt.ylim(0, top_y)
    except ValueError:
        print('bad input--did it converge?')
    if symmetric:
        plt.xlim(0, 55)
    else:
        plt.xlim(-55, 55)

    if theory is not None:
        Y = [c for c in theory.columns if c != 'lat']
        if symmetric:
            nrows = theory.shape[0]
            midpoint = int(nrows/2)
            x_plot = theory['lat'][midpoint:]
            theory_sym = pd.DataFrame(columns=theory.columns)
            theory_sym['lat'] = x_plot
            theory_sym[Y] = np.array(theory[Y][midpoint:])+np.flipud(np.array(theory[Y][:midpoint]))
            theory_sym[Y] /= 2
            theory = theory_sym
        else:
            x_plot = theory['lat']
        if theory.shape[1] == 2:
            plt.plot(x_plot, theory.iloc[:, 1], label='Theoretical')
        elif theory.shape[1] == 6:
            for vel in Y:
                plt.plot(x_plot, theory[vel], label=f'{vel}km/s radiant')

    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    line = Line2D([0], [0], label='Posterior samples', color='black')
    line2 = Line2D([0], [0], label='Central 80\%', color='red', linestyle='--')
    line3 = Line2D([0], [0], label='Median', color='red', linestyle='-')
    if plot_map:
        line4 = Line2D([0], [0], label='MAP', color='red', linestyle='-')
        handles.extend([line, line2, line3, line4])
    else:
        handles.extend([line, line2, line3])

    plt.legend(handles=handles, frameon=False, ncol=2)
    plt.gcf().set_size_inches((4, 3))
    # plt.savefig(f'plots/{filename}.png', dpi=300, bbox_inches='tight')
    # plt.savefig(f'plots/{filename}.pgf', bbox_inches='tight')
    plt.savefig(f'plots/{filename}.pdf', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_fov_result(result, title, filename, plot_map=True, normalize=False, angle=False, truth=None, show=False):
    plt.style.use('default')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, ax1 = plt.subplots()
    ax1.xaxis.set_tick_params(which='major', top='on')

    f_fov = result['f_lat']
    result = result['results'][0]
    pos = result['idata'].posterior
    m_ap = result['map']
    max_area = result['max_area']

    top_y = 0
    y_plots = np.zeros((400, 200))
    for i in range(401):
        x_plot = np.linspace(0, 7300, 200)/10000
        chain = random.randint(0, len(pos.chain)-1)
        draw = random.randint(0, len(pos.draw)-1)

        if i != 400:
            color = 'black'
            alpha = 0.2
            linewidth = 0.5

            intercept = pos.intercept[chain][draw].data
            varfactor = get_varfactor(f_fov, 'fov', x_plot, pos=pos, chain=chain, draw=draw)

        else:
            color = 'red'
            alpha = 1
            linewidth = 1

            # plot MAP
            intercept = m_ap['intercept']
            varfactor = get_varfactor(f_fov, 'fov', x_plot, m_ap=m_ap)

        if normalize:
            y_plot = np.exp(varfactor)
        else:
            y_plot = np.exp(intercept+varfactor)/max_area

        top_y = max(max(y_plot), top_y)

        if angle:
            x_plot = [distance_to_angle(x*10000) for x in x_plot]

        if i != 400:
            y_plots[i, :] = y_plot
        if not (i==400 and not plot_map):
            ax1.plot(x_plot, y_plot, color=color, alpha=alpha, linewidth=linewidth)

    top_quantile = np.quantile(y_plots, 0.90, axis=0)
    bottom_quantile = np.quantile(y_plots, 0.10, axis=0)
    median = np.quantile(y_plots, 0.5, axis=0)
    ax1.plot(x_plot, top_quantile, color='red', linestyle='--', linewidth=1)
    ax1.plot(x_plot, bottom_quantile, color='red', linestyle='--', linewidth=1)
    plt.plot(x_plot, median, color='red', linewidth=1, linestyle='-')

    if truth is not None and angle:
        ax2 = ax1.twinx()
        x = truth['AOI']
        bg = truth['Background']
        raw = truth['Lightning']
        ax2.plot(x, bg, label="Per background")
        ax2.plot(x, raw, label="Raw signal")
        ax2.set_ylim(0, 1.1*max(max(bg), max(raw)))
        ax2.set_ylabel('Fraction of lightning signal reaching detector')

    if angle:
        ax1.set_xlabel('Angle of incident light upon sensor (°)')
    else:
        ax1.set_xlabel('distance from fov nadir (km)')
    ax1.set_ylim(0, top_y)
    plt.xlim(0, max(x_plot))
    ax1.set_ylabel('Bolide rate')
    plt.title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    line = Line2D([0], [0], label='Posterior samples', color='black')
    line2 = Line2D([0], [0], label='Central 80\%', color='red', linestyle='--')
    line3 = Line2D([0], [0], label='Median', color='red', linestyle='-')
    if plot_map:
        line4 = Line2D([0], [0], label='MAP', color='red', linestyle='-')
        handles.extend([line, line2, line3, line4])
    else:
        handles.extend([line, line2, line3])
    plt.legend(handles=handles, frameon=False)
    plt.gcf().set_size_inches((4, 3))
    plt.savefig(f'plots/{filename}.pdf', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_pitch_result(result, title, filename, normalize=False):

    pos = result.posterior
    for i in range(400):
        chain = random.randint(0, len(pos.chain)-1)
        draw = random.randint(0, len(pos.draw)-1)

        b0 = pos.Intercept[chain][draw].data
        b8 = pos.pitch[chain][draw].data
        b9 = pos.pitch2[chain][draw].data

        x_plot = np.linspace(600, 900)/900
        if normalize:
            y_plot = [np.exp(b8 * x + b9 * x**2) for x in x_plot]
        else:
            y_plot = [np.exp(b0+b8 * x + b9 * x**2) for x in x_plot]
        plt.plot(x_plot*900, y_plot, color='lightblue', alpha=0.05)

    plt.xlabel('pixel area (microns$^2$)')
    plt.ylabel('Bolide rate')
    plt.title(title+' in 400 NUTS samples')
    plt.savefig(f'plots/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()


def get_pixel_from_dist(dist, dist_to_pixel):
    keys = np.array(list(dist_to_pixel.keys()))
    idx = np.abs(keys-dist).argmin()
    key = keys[idx]
    return dist_to_pixel[key]


def plot_fov_result_pixel(result, title, filename, dist_to_pixel):

    pos = result.posterior
    for i in range(400):
        chain = random.randint(0, len(pos.chain)-1)
        draw = random.randint(0, len(pos.draw)-1)

        # intercept = pos.Intercept[0][1].data
        b0 = pos.Intercept[chain][draw].data
        b3 = pos.fov_dist[chain][draw].data
        b4 = pos.fov_dist2[chain][draw].data
        b5 = pos.fov_dist3[chain][draw].data

        x_plot = np.linspace(0, 7300, 200)/10000
        y_plot = [np.exp(b0+b3 * x + b4 * x**2 + b5 * x**3) for x in x_plot]

        x_plot_pixel = [get_pixel_from_dist(x*10000, dist_to_pixel) for x in x_plot]

        plt.plot(x_plot_pixel, y_plot, color='lightblue', alpha=0.05)

    plt.xlabel('distance from fov center (pixels)')
    plt.ylabel('Bolide rate')
    plt.title(title+' in 400 NUTS samples')
    plt.savefig(f'plots/{filename}.png', dpi=300, bbox_inches='tight')
    plt.close()
