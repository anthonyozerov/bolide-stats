import matplotlib.pyplot as plt
import numpy as np
import random

from matplotlib.lines import Line2D

from .geo_utils import distance_to_angle


def plot_lat_result(result, title, filename, max_lat=55, normalize=True, symmetric=False, shower='', theory=None):
    plt.style.use('default')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, ax1 = plt.subplots()
    ax1.xaxis.set_tick_params(which='major', top='on')
    ax1.yaxis.set_tick_params(which='major', top='on')

    pos = result['posterior'].posterior
    m_ap = result['map']
    top_y = 0
    y_plots = np.zeros((400, 200//(int(symmetric)+1)))
    for i in range(400+1):
        x_plot = np.linspace(-max_lat, max_lat, 200)/90
        chain = random.randint(0, len(pos.chain)-1)
        draw = random.randint(0, len(pos.draw)-1)

        if i != 400:
            color = 'black'
            alpha = 0.2
            linewidth = 0.5

            intercept = pos.__getattr__(shower+"intercept")[chain][draw].data
            l1 = pos.__getattr__(shower+"lat1")[chain][draw].data
            l2 = pos.__getattr__(shower+"lat2")[chain][draw].data
            l3 = pos.__getattr__(shower+"lat3")[chain][draw].data
        else:
            color = 'red'
            alpha = 1
            linewidth = 1

            # plot MAP
            intercept = m_ap['intercept']
            l1 = m_ap[shower+"lat1"]
            l2 = m_ap[shower+"lat2"]
            l3 = m_ap[shower+"lat3"]

        if normalize:
            y_plot = [np.exp(l1*x + l2*x**2 + l3*np.abs(x**3)) for x in x_plot]
        else:
            y_plot = [np.exp(intercept + l1*x + l2*x**2 + l3*np.abs(x**3)) for x in x_plot]
        y_plot = np.array(y_plot)
        if symmetric:
            midpoint = int(len(x_plot)/2)
            x_plot = x_plot[midpoint:]
            y_plot = (y_plot[midpoint:] + np.flipud(y_plot[:midpoint]))/2
        top_y = max(max(y_plot), top_y)
        if i != 400:
            y_plots[i, :] = y_plot
        ax1.plot(x_plot*90, y_plot, color=color, alpha=alpha, linewidth=linewidth)

    # plot quantiles
    top_quantile = np.quantile(y_plots, 0.90, axis=0)
    bottom_quantile = np.quantile(y_plots, 0.10, axis=0)
    plt.plot(x_plot*90, top_quantile, color='red', linewidth=1, linestyle='--')
    plt.plot(x_plot*90, bottom_quantile, color='red', linewidth=1, linestyle='--')

    if normalize:
        y_plot = [np.exp(l1*x + l2*x**2 + l3*np.abs(x**3)) for x in x_plot]
    else:
        y_plot = [np.exp(intercept + l1*x + l2*x**2 + l3*np.abs(x**3)) for x in x_plot]
    y_plot = np.array(y_plot)
    if symmetric:
        midpoint = int(len(x_plot)/2)
        x_plot = x_plot[midpoint:]
        y_plot = (y_plot[midpoint:] + np.flipud(y_plot[:midpoint]))/2

    plt.xlabel('Latitude (°)')
    plt.ylabel('Normalized bolide flux')
    plt.ylim(0, top_y)
    if symmetric:
        plt.xlim(0, 55)
    else:
        plt.xlim(-55, 55)
    if shower == 'LEO':
        shower = 'Leonid'

    if theory is not None:
        if symmetric:
            nrows = theory.shape[0]
            midpoint = int(nrows/2)
            x_plot = theory[midpoint:, 0]
            theory = theory[midpoint:, :]+np.flipud(theory)[midpoint:, :]
            theory /= 2
        else:
            x_plot = theory[:, 0]
        if theory.shape[1] == 2:
            plt.plot(x_plot, theory[:, 1], label='Theoretical')
        elif theory.shape[1] == 6:
            for i, vel in enumerate([50, 55, 60, 65, 68]):
                plt.plot(x_plot, theory[:, i+1], label=f'Theoretical, {vel}km/s radiant')

    plt.title(shower+' '+title)
    handles, labels = plt.gca().get_legend_handles_labels()
    line = Line2D([0], [0], label='Posterior bolide rate samples', color='black')
    line2 = Line2D([0], [0], label='Central 80% of distribution', color='red', linestyle='--')
    line3 = Line2D([0], [0], label='MAP', color='red', linestyle='-')
    handles.extend([line, line2, line3])
    plt.legend(handles=handles, frameon=False)
    plt.gcf().set_size_inches((4, 3))
    plt.savefig(f'posteriors/{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'posteriors/{filename}.pgf', bbox_inches='tight')
    plt.savefig(f'posteriors/{filename}.pdf', bbox_inches='tight')
    plt.show()


def plot_fov_result(result, title, filename, normalize=False, angle=False, truth=None):
    plt.style.use('default')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

    fig, ax1 = plt.subplots()
    ax1.xaxis.set_tick_params(which='major', top='on')

    pos = result['posterior'].posterior
    m_ap = result['map']
    top_y = 0
    y_plots = np.zeros((400, 200))
    for i in range(400+1):
        chain = random.randint(0, len(pos.chain)-1)
        draw = random.randint(0, len(pos.draw)-1)

        if i != 400:
            color = 'black'
            alpha = 0.2
            linewidth = 0.5

            intercept = pos.intercept[chain][draw].data
            fov1 = pos.fov_dist[chain][draw].data
            fov2 = pos.fov_dist2[chain][draw].data
            fov3 = pos.fov_dist3[chain][draw].data

        else:
            color = 'red'
            alpha = 1
            linewidth = 1

            # plot MAP
            intercept = m_ap['intercept']
            fov1 = m_ap["fov_dist"]
            fov2 = m_ap["fov_dist2"]
            fov3 = m_ap["fov_dist3"]

        x_plot = np.linspace(0, 7300, 200)/10000
        # x_plot = np.linspace(0,9,200)/10
        if normalize:
            y_plot = [np.exp(fov1*x + fov2*x**2 + fov3*x**3) for x in x_plot]
        else:
            y_plot = [np.exp(intercept + fov1*x + fov2*x**2 + fov3*x**3) for x in x_plot]

        top_y = max(max(y_plot), top_y)

        x_plot *= 10000
        if angle:
            x_plot = [distance_to_angle(x) for x in x_plot]

        if i != 400:
            y_plots[i, :] = y_plot
        ax1.plot(x_plot, y_plot, color=color, alpha=alpha, linewidth=linewidth)

    top_quantile = np.quantile(y_plots, 0.90, axis=0)
    bottom_quantile = np.quantile(y_plots, 0.10, axis=0)
    ax1.plot(x_plot, top_quantile, color='red', linestyle='--', linewidth=1)
    ax1.plot(x_plot, bottom_quantile, color='red', linestyle='--', linewidth=1)
    #plt.plot(x_plot*10, top_quantile, color='lightblue')
    #plt.plot(x_plot*10, bottom_quantile, color='lightblue')
    if truth is not None and angle:
        ax2 = ax1.twinx()
        x = truth['x']
        bg = truth['bg']
        raw = truth['raw']
        ax2.plot(x, bg, label="Per background")
        ax2.plot(x, raw, label="Raw signal")
        ax2.set_ylim(0, 1)
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
    line = Line2D([0], [0], label='Posterior bolide rate samples', color='black')
    line2 = Line2D([0], [0], label='Central 80\% of distribution', color='red', linestyle='--')
    line3 = Line2D([0], [0], label='MAP', color='red', linestyle='-')
    handles.extend([line, line2, line3])
    plt.legend(handles=handles, frameon=False)
    plt.gcf().set_size_inches((8, 3))
    plt.savefig(f'posteriors/{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'posteriors/{filename}.pgf', bbox_inches='tight')
    plt.savefig(f'posteriors/{filename}.pdf', bbox_inches='tight')
    plt.show()


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
    plt.savefig(f'posteriors/{filename}.png', dpi=300, bbox_inches='tight')


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
    plt.savefig(f'posteriors/{filename}.png', dpi=300, bbox_inches='tight')
