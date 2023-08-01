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

from bolides.constants import GLM_STEREO_MIDPOINT
import bolides.crs as bcrs


def plot_polygons(gdf, data, filename=None, label='Polygon area (km$^{-2}$)',
                  crs=ccrs.EckertIV(central_longitude=GLM_STEREO_MIDPOINT), second='insert',
                  crs2=bcrs.GOES_E(), show=False, figsize=(10, 4), extent=None):

    plt.style.use('default')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    crs_proj4 = crs.proj4_init
    gdf = gdf.to_crs(crs_proj4)
    if second == 'side':
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 2, 1, projection=crs)
        ax2 = fig.add_subplot(1, 2, 2, projection=crs2)
        plt.tight_layout()
    else:
        fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize=figsize)
    ax.stock_img()
    cmap = plt.get_cmap('viridis')

    cNorm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    colors = [scalarMap.to_rgba(num) for num in data]
    for num, poly in enumerate(gdf.geometry):
        ax.add_geometries([poly], crs=crs, color=colors[num], linewidth=0)

    points = [Point(lon, lat) for lon, lat in zip(gdf['lon'], gdf['lat'])]
    points = GeoDataFrame(geometry=points, crs='epsg:4326').to_crs(crs.proj4_init).geometry
    x = np.array([p.x for p in points])
    y = np.array([p.y for p in points])
    ax.scatter(x, y, color='red', marker='.', s=4, zorder=5, edgecolor='none')

    if second == 'inset':
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import cartopy
        ax2 = inset_axes(ax, width="100%", height="100%", loc='lower left',
                         bbox_to_anchor=(0.5, -0.05, 0.75, 0.6),
                         bbox_transform=ax.transAxes,
                         axes_class=cartopy.mpl.geoaxes.GeoAxes,
                         axes_kwargs=dict(map_projection=crs2))
    if second == 'side':
        plt.colorbar(scalarMap, label=label, ax=[ax, ax2])
    else:
        plt.colorbar(scalarMap, label=label)
    if second is not None:
        ax2.stock_img()
        for num, poly in enumerate(gdf.geometry):
            ax2.add_geometries([poly], crs=crs, color=colors[num], linewidth=0)
        points = [Point(lon, lat) for lon, lat in zip(gdf['lon'], gdf['lat'])]
        points = GeoDataFrame(geometry=points, crs='epsg:4326').to_crs(crs2.proj4_init).geometry
        x = np.array([p.x for p in points])
        y = np.array([p.y for p in points])
        plt.scatter(x, y, color='red', marker='.', s=3, zorder=5, edgecolor='none')
        ax2.coastlines()
        ax2.gridlines(color='black', draw_labels=False, xlocs=[0, 60, 120, 180, -60, -120])

    ax.coastlines()
    ax.gridlines(color='black', draw_labels=(second != 'side'))

    if extent is not None:
        ax.set_extent(extent, ccrs.PlateCarree())

    if filename is not None:
        plt.savefig(f'data-plots/{filename}.pdf', bbox_inches='tight')
    if show:
        plt.show()
    # gdf.plot(column)
    # plt.show()


def reformat_result(result, which=0):
    import copy
    r = result['results'][which]
    pos = r['idata'].posterior
    pos2 = {}
    for key in pos.keys():
        pos2[key] = pos[key]
    for i, var in enumerate(pos.beta.predictors):
        varname = str(np.array(var))
        pos2[varname] = pos.beta[:, :, i]

    pos2['chain'] = pos.chain
    pos2['draw'] = pos.draw
    result2 = copy.deepcopy(result)
    result2['results'][which]['idata'] = {'posterior': pos2}
    return result2


def get_varfactor(f, var, x, shower='', adjust=None, m_ap=None, pos=None, chain=None, draw=None, start_idx=None):
    assert m_ap is not None or pos is not None
    if pos is not None:
        assert chain is not None and draw is not None

    zero = 0
    varfactor = np.zeros(len(x))
    for i, term in enumerate(f.split('+')):
        power = int(term.split('^')[1])
        key = shower+var+str(power)

        if m_ap is not None:
            varfactor += m_ap['beta'][start_idx+i] * ((x**power)-adjust['mean'][key])/adjust['scale'][key]
            zero -= m_ap['beta'][start_idx+i]*adjust['mean'][key]/adjust['scale'][key]
        elif pos is not None:
            varfactor += pos[key][chain][draw].data * ((x**power)-adjust['mean'][key])/adjust['scale'][key]
            zero -= pos[key][chain][draw].data*adjust['mean'][key]/adjust['scale'][key]
    return varfactor, zero


def plot_lat_result(result, title, which=0, filename=None, to_plot=['samples', 'median', 'quantiles'],
                    max_lat=55, normalize=True, symmetric=False, shower='', theory=None, show=False,
                    legend_col=2, figsize=(4, 3), ylim=None):
    result = reformat_result(result, which=which)

    plt.style.use('default')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, ax1 = plt.subplots()
    ax1.xaxis.set_tick_params(which='major', top='on')
    ax1.yaxis.set_tick_params(which='major', top='on')

    f_lat = result['f_lat']
    pos = result['results'][which]['idata']['posterior']
    m_ap = result['results'][0]['map']
    max_area = result['results'][which]['max_area']

    top_y = 0
    y_plots = np.zeros((400, 200//(int(symmetric)+1)))
    x_plot = np.linspace(-max_lat, max_lat, 200)
    for i in range(400):
        chain = random.randint(0, len(pos['chain'])-1)
        draw = random.randint(0, len(pos['draw'])-1)

        color = 'black'
        alpha = 0.2
        linewidth = 0.5

        intercept = pos[f"{shower}intercept"][chain][draw].data
        adjust = result['results'][which]['adjust']
        varfactor, zero = get_varfactor(f_lat, 'lat', x_plot, shower, adjust, pos=pos, chain=chain, draw=draw)
        if normalize:
            y_plot = np.exp(varfactor)/np.exp(zero)
        else:
            y_plot = np.exp(intercept + varfactor)/max_area
        y_plot = np.array(y_plot)
        if symmetric:
            midpoint = int(len(x_plot)/2)
            y_plot = (y_plot[midpoint:] + np.flipud(y_plot[:midpoint]))/2
        else:
            midpoint = 0
        top_y = max(max(y_plot), top_y)
        y_plots[i, :] = y_plot
        if 'samples' in to_plot:
            ax1.plot(x_plot[midpoint:], y_plot, color=color, alpha=alpha, linewidth=linewidth)

    if 'map' in to_plot:
        color = 'red'
        alpha = 1
        linewidth = 1

        intercept = m_ap[f"{shower}intercept"]
        adjust = result['results'][which]['adjust']
        varfactor, zero = get_varfactor(f_lat, 'lat', x_plot, shower, adjust, m_ap=m_ap, start_idx=0)

        if normalize:
            y_plot = np.exp(varfactor)/np.exp(zero)
        else:
            y_plot = np.exp(intercept + varfactor)/max_area
        y_plot = np.array(y_plot)
        if symmetric:
            midpoint = int(len(x_plot)/2)
            y_plot = (y_plot[midpoint:] + np.flipud(y_plot[:midpoint]))/2
        else:
            midpoint = 0
        ax1.plot(x_plot[midpoint:], y_plot, color=color, alpha=alpha, linewidth=linewidth)

    # plot quantiles
    top_quantile = np.quantile(y_plots, 0.90, axis=0)
    bottom_quantile = np.quantile(y_plots, 0.10, axis=0)
    median = np.quantile(y_plots, 0.5, axis=0)
    if 'median' in to_plot:
        plt.plot(x_plot[midpoint:], median, color='red', linewidth=1, linestyle='-')
        plt.ylim(0, max(median)*1.1)
    if 'quantiles' in to_plot:
        plt.plot(x_plot[midpoint:], top_quantile, color='red', linewidth=1, linestyle='--')
        plt.plot(x_plot[midpoint:], bottom_quantile, color='red', linewidth=1, linestyle='--')
        plt.ylim(0, max(top_quantile)*1.1)
    if ylim is not None:
        plt.ylim(0, ylim)

    plt.xlabel('Latitude (°)')
    plt.ylabel('Normalized bolide flux')
    # try:
    #     plt.ylim(0, top_y)
    # except ValueError:
    #     print('bad input--did it converge?')
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
        else:
            for vel in ['50', '60', '68']:
                plt.plot(x_plot, theory[vel], label=f'{vel}km/s radiant')
            plt.plot(x_plot, theory['all'], label='All radiants')

    plt.title(title)
    handles, labels = plt.gca().get_legend_handles_labels()
    if 'samples' in to_plot:
        handles.extend([Line2D([0], [0], label='Posterior samples', color='black')])
    if 'quantiles' in to_plot:
        handles.extend([Line2D([0], [0], label=r'Central 80\%', color='red', linestyle='--')])
    if 'median' in to_plot:
        handles.extend([Line2D([0], [0], label='Median', color='red', linestyle='-')])
    if 'map' in to_plot:
        handles.extend([Line2D([0], [0], label='MAP', color='red', linestyle='-')])

    plt.legend(handles=handles, frameon=False, ncol=legend_col)
    plt.gcf().set_size_inches(figsize)
    # plt.savefig(f'plots/{filename}.png', dpi=300, bbox_inches='tight')
    # plt.savefig(f'plots/{filename}.pgf', bbox_inches='tight')
    if filename is not None:
        plt.savefig(f'plots/{filename}.pdf', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_fov_result(result, title, filename=None, plot_map=True, normalize=False, angle=False,
                    truth=None, show=False, figsize=(4, 3)):
    result = reformat_result(result)

    plt.style.use('default')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, ax1 = plt.subplots()
    ax1.xaxis.set_tick_params(which='major', top='on')

    f_fov = result['f_fov']
    pos = result['results'][0]['idata']['posterior']
    m_ap = result['results'][0]['map']
    max_area = result['results'][0]['max_area']

    top_y = 0
    y_plots = np.zeros((400, 200))
    for i in range(401):
        x_plot = np.linspace(0, 8.3, 200)
        chain = random.randint(0, len(pos['chain'])-1)
        draw = random.randint(0, len(pos['draw'])-1)

        if i != 400:
            color = 'black'
            alpha = 0.2
            linewidth = 0.5

            adjust = result['results'][0]['adjust']
            intercept = pos['intercept']
            varfactor, zero = get_varfactor(f_fov, 'fov', x_plot, adjust=adjust, pos=pos, chain=chain, draw=draw)

        elif plot_map:
            color = 'red'
            alpha = 1
            linewidth = 1

            # plot MAP
            intercept = m_ap['intercept']
            varfactor = get_varfactor(f_fov, 'fov', x_plot, m_ap=m_ap)

        if normalize:
            y_plot = np.exp(varfactor)/np.exp(zero)
        else:
            y_plot = np.exp(intercept+varfactor)/max_area

        top_y = max(max(y_plot), top_y)

        if i != 400:
            y_plots[i, :] = y_plot
        if not (i == 400 and not plot_map):
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
        ax2.plot(x, bg, label="Integrated GLM background brightness")
        ax2.plot(x, raw, label="Fraction of lightning signal reaching GLM")
        ax2.set_ylim(0, 1.1*max(max(bg), max(raw)))
        ax2.set_ylabel('Measured GLM sensitivity')

    if angle:
        ax1.set_xlabel('Angle of incident light upon sensor (°)')
    else:
        ax1.set_xlabel('distance from fov nadir (km)')
    ax1.set_ylim(0, top_y)
    plt.xlim(0, max(x_plot))
    ax1.set_ylabel('Normalized bolide flux')
    plt.title(title)

    handles, labels = plt.gca().get_legend_handles_labels()
    line = Line2D([0], [0], label='Posterior samples', color='black')
    line2 = Line2D([0], [0], label=r'Central 80\%', color='red', linestyle='--')
    line3 = Line2D([0], [0], label='Median', color='red', linestyle='-')
    if plot_map:
        line4 = Line2D([0], [0], label='MAP', color='red', linestyle='-')
        handles.extend([line, line2, line3, line4])
    else:
        handles.extend([line, line2, line3])
    plt.legend(handles=handles, frameon=False)
    plt.gcf().set_size_inches(figsize)
    if filename is not None:
        plt.savefig(f'plots/{filename}.pdf', bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_fov_results(results, title, angle=True, truth=None, figsize=(4, 3)):

    plt.style.use('default')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['ytick.right'] = True

    fig, ax1 = plt.subplots()
    fig.set_size_inches(figsize)

    colors = ['darkblue', 'brown']

    for n, result in enumerate(results):
        result = reformat_result(result)
        f_fov = result['f_lat']
        pos = result['results'][0]['idata']['posterior']
        # m_ap = result['results'][0]['map']
        # max_area = result['results'][0]['max_area']

        y_plots = np.zeros((400, 200))
        for i in range(400):
            x_plot = np.linspace(0, 8.3, 200)
            chain = random.randint(0, len(pos['chain'])-1)
            draw = random.randint(0, len(pos['draw'])-1)

            adjust = result['results'][0]['adjust']
            varfactor, zero = get_varfactor(f_fov, 'fov', x_plot, adjust=adjust, pos=pos, chain=chain, draw=draw)

            y_plot = np.exp(varfactor)/np.exp(zero)

            y_plots[i, :] = y_plot

        top_quantile = np.quantile(y_plots, 0.90, axis=0)
        bottom_quantile = np.quantile(y_plots, 0.10, axis=0)
        median = np.quantile(y_plots, 0.5, axis=0)
        ax1.plot(x_plot, top_quantile, color=colors[n], linestyle='--', linewidth=0.5)
        ax1.plot(x_plot, bottom_quantile, color=colors[n], linestyle='--', linewidth=0.5)
        plt.plot(x_plot, median, color=colors[n], linewidth=1, linestyle='-')

    if truth is not None and angle:
        ax2 = ax1.twinx()
        x = truth['AOI']
        bg = truth['Background']
        raw = truth['Lightning']
        ax2.plot(x, bg, color='black', label="Integrated GLM background brightness", linestyle=':')
        ax2.plot(x, raw, color='gray', label="Fraction of lightning signal reaching GLM", linestyle=':')
        ax2.set_ylim(0, 1.1*max(max(bg), max(raw)))
        ax2.set_ylabel('Measured GLM sensitivity')

    ax1.set_xlabel('Angle of incident light upon sensor (°)')

    plt.xlim(0, max(x_plot))
    ax1.set_ylabel('Normalized bolide flux')
    plt.title(title)

    legend1 = plt.legend(loc=4, frameon=False, title=r'\textbf{Right Axis}')

    handles, labels = plt.gca().get_legend_handles_labels()
    lines = []
    lines.append(Line2D([0], [0], label=r'Central 80\% (\texttt{human})', color=colors[0], linestyle='--'))
    lines.append(Line2D([0], [0], label=r'Median (\texttt{human})', color=colors[0], linestyle='-'))
    lines.append(Line2D([0], [0], label=r'Central 80\% (\texttt{machine})', color=colors[1], linestyle='--'))
    lines.append(Line2D([0], [0], label=r'Median (\texttt{machine})', color=colors[1], linestyle='-'))

    labels = [r'Central 80\% (\texttt{human})\ \ \ \textbf{Left Axis}',
              r'Median (\texttt{human})',
              r'Central 80\% (\texttt{machine})',
              r'Median (\texttt{machine})']

    handles.extend(lines)

    plt.legend(handles[2:], labels, loc=3, frameon=False)
    plt.gca().add_artist(legend1)

    # plt.legend(handles=handles, frameon=False, loc='lower center', ncol=2)
