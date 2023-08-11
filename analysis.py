# This file generates a lot of the figures in the paper, and a lot of extras
import pickle
from plotting import plot_lat_result, plot_fov_result, plot_fov_results, plot_polygons, contourplot
import arviz as az
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import os
from bolides.constants import GOES_E_LON
from corner import corner

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.right'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.major.top'] = True
plt.rcParams['ytick.major.right'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['text.usetex'] = True

os.environ['PATH'] += ':/home/aozerov/.texlive/2023/bin/x86_64-linux'

if not os.path.exists('plots'):
    os.makedirs('plots')

goes_e_eck4 = ccrs.EckertIV(central_longitude=GOES_E_LON)


def csv_else_none(filename, skiprows=0, sep=','):
    if os.path.exists(filename):
        return pd.read_csv(filename, skiprows=skiprows, sep=sep)
    return None


lat_truth = csv_else_none('impacts/impact-dists.csv')
fov_truth = csv_else_none('data/glm-bandpass.txt', skiprows=9, sep='\t')
leo_truth = csv_else_none('impacts/leonids-dists.csv')
per_truth = csv_else_none('impacts/perseids-dists.csv')


def load_model(name):
    print('------------')
    print(name)
    print('------------')
    with open(f'models/{name}.pkl', 'rb') as f:
        result = pickle.load(f)
    pos = result['results'][0]['idata'].posterior
    adjust = result['results'][0]['adjust']

    print('ArviZ summary:')
    # Check that ess_bulk and ess_tail in the summary are in the thousands,
    # and r_hat is <= 1.01. This indicates that the sampler converged.
    print(az.summary(pos))

    return result, pos, adjust


# PAPER MODEL 1
result, pos, adjust = load_model('reg')
pos = result['results'][0]['idata'].posterior
adjust = result['results'][0]['adjust']

plt.figure(figsize=(4, 3))
plt.hist(np.exp(np.array(pos.beta[:, :, 1]).flatten()/adjust['scale']['land']),
         bins=200, density=True, histtype='step', color='black', linewidth=2)
plt.xlabel(r'$e^{\gamma_{\mathrm{land}}}$ (Land detection efficiency multiplier)')
plt.ylabel('Posterior density')
plt.savefig('plots/machine-land.pdf', bbox_inches='tight')
plt.close()

plt.figure(figsize=(4, 3))
plt.hist(np.exp(np.array(pos.beta[:, :, 2]).flatten()/adjust['scale']['cloud_prop']),
         bins=200, density=True, histtype='step', color='black', linewidth=2)
plt.xlabel(r'$e^{\gamma_{\mathrm{cloud}}}$ (Cloud detection efficiency multiplier)')
plt.ylabel('Posterior density')
plt.savefig('plots/machine-cloud.pdf', bbox_inches='tight')
plt.close()

flash = np.exp(np.array(pos.beta[:, :, 0]).flatten()/adjust['scale']['flash_dens'])
land = np.exp(np.array(pos.beta[:, :, 1]).flatten()/adjust['scale']['land'])
cloud = np.exp(np.array(pos.beta[:, :, 2]).flatten()/adjust['scale']['cloud_prop'])

contourplot(flash, land, 'flash', 'land', 'biases1')
contourplot(flash, cloud, 'flash', 'cloud', 'biases2')
contourplot(cloud, land, 'cloud', 'land', 'biases3')

plot_lat_result(result, filename='machine-lat', title=None,
                show=False, normalize=True, symmetric=False, theory=lat_truth, ylim=1.3)
plot_lat_result(result, filename='machine-lat-sym', title=None,
                show=False, normalize=True, symmetric=True, theory=lat_truth, ylim=1.3)
plot_fov_result(result, filename='machine-fov',
                title=r'Bolide rate dependent on angle of incidence in \texttt{machine}',
                show=False, plot_map=None, normalize=True, angle=True, figsize=(4, 3), truth=fov_truth)
plot_fov_result(result, filename='machine-fov-acm', title=r'Bolide rate dependent on angle of incidence',
                show=False, plot_map=None, normalize=True, angle=True, figsize=(8, 3), truth=fov_truth)
fig = corner(pos, var_names=['beta', 'intercept'])
for ax in fig.axes:
    ax.ylabel = None
    ax.xlabel = None
    ax.xticks = False
    ax.yticks = False
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
fig.savefig('plots/machine-corner.pdf')

# PAPER MODEL 2
# The reference result for human-vetted data.
result, pos, adjust = load_model('human-reg-S')

plt.figure(figsize=(4, 3))
plt.hist(np.exp(np.array(pos.beta[:, :, 2]).flatten()/adjust['scale']['stereo']),
         bins=200, density=True, histtype='step', color='black', linewidth=2)
plt.xlabel(r'$e^{\gamma_{\mathrm{stereo}}}$ (Stereo region detection efficiency multiplier)')
plt.ylabel('Posterior density')
plt.savefig('plots/human-stereo.pdf', bbox_inches='tight')

plot_lat_result(result, filename='human-lat', title=None,
                show=False, normalize=True, symmetric=False, theory=lat_truth, ylim=1.3)
plot_lat_result(result, filename='human-lat-sym', title=None,
                show=False, normalize=True, symmetric=True, theory=lat_truth, ylim=1.3)
plot_fov_result(result, filename='human-fov', title=None,
                show=False, plot_map=False, normalize=True, angle=True, figsize=(8, 3), truth=fov_truth)

# joint plot of AOI bias in Model 1 and Model 2
r1, _, _ = load_model('human-reg-S')
r2, _, _ = load_model('reg')
plot_fov_results([r1, r2], title=r'AOI bias in Model 1 (\texttt{machine}) and Model 2 (\texttt{human})',
                 truth=fov_truth, figsize=(8, 3))
plt.savefig('plots/fov-machine-human.pdf', bbox_inches='tight')

# PAPER MODELS 3 AND 4
sats = ['GOES-16', 'GOES-17']
sats_short = ['g16', 'g17']
result, _, _ = load_model('reg-separate')
for i in range(2):
    plot_lat_result(result, filename=f'reg-separate-{sats_short[i]}', which=i, title=sats[i],
                    show=False, normalize=True, symmetric=False, theory=lat_truth, ylim=1.3)

# PAPER MODEL 5
result, pos, adjust = load_model('leoper')
plot_lat_result(result, title='Leonids', filename='leoper-leo-lat',
                show=False, normalize=True, symmetric=False,
                shower='LEO', theory=leo_truth, legend_col=1)
plot_lat_result(result, title='Perseids', filename='leoper-per-lat',
                show=False, normalize=True, symmetric=False,
                shower='PER', theory=per_truth, legend_col=1)

# PAPER MODEL 6
result, pos, adjust = load_model('leoper-known')
plot_lat_result(result, filename='leoper-known-lat', title=None,
                show=False, normalize=True, symmetric=False, theory=lat_truth, ylim=1.3)
plot_lat_result(result, filename='leoper-known-lat-sym', title=None,
                show=False, normalize=True, symmetric=True, theory=lat_truth, ylim=1.3)
plot_fov_result(result, filename='leoper-known-fov', title=r'Bolide rate dependent on angle of incidence',
                show=False, plot_map=None, normalize=True, angle=True, figsize=(8, 3), truth=fov_truth)

# PAPER MODELS 7-12
for conf in [0.25, 0.5, 0.8, 0.9, 0.95, 0.99]:
    result, _, _ = load_model(f'reg-{conf}')
    plot_lat_result(result, filename=f'reg-{conf}-lat', title=conf,
                    show=False, normalize=True, symmetric=False, theory=lat_truth, ylim=1.3)
