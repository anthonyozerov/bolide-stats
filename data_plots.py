# This script produces various informative plots ancillary to the main analysis

import matplotlib.pyplot as plt
import os
from bolides import BolideDataFrame
import numpy as np
import pandas as pd
from tqdm import tqdm
import cartopy.crs as ccrs
from bolides.constants import GLM_STEREO_MIDPOINT, GOES_E_LON
from haversine import haversine_vector
from bolides.fov_utils import get_boundary
from shapely.geometry import Polygon
from scipy.stats.distributions import chi2
from astropy.coordinates import get_sun
from astropy.time import Time
from sklearn.neighbors import KernelDensity
from plotting import plot_polygons
Eck4 = ccrs.EckertIV(central_longitude=GLM_STEREO_MIDPOINT)

plt.style.use('default')
plt.rcParams['text.usetex'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.top'] = True
plt.rcParams['ytick.major.right'] = True
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True

bdf = BolideDataFrame(source='csv', files='data/pipeline-dedup.csv', annotate=False)
bdf = bdf[bdf.confidence > 0.7]

##########################
# MAIN DATA PLOT
fig, ax = bdf.plot_detections(category='detectedBy', boundary=['goes-e', 'goes-w'], background=True, coastlines=True,
                              s=5, marker='.', figsize=(10, 4), colors={'G16': 'brown', 'G17': 'darkblue'}, crs=Eck4)
ax.gridlines(color='black', draw_labels=True)
fig.savefig('plots/detections.pdf', bbox_inches='tight')
plt.close()

###########################
# HUMAN VS MACHINE PLOT
bdf_train = BolideDataFrame()
sat_pipe = ['G16', 'G17']
sat_web = ['GLM-16', 'GLM-17']
fp = []
fn = []
both = 0
# do it for each satellite separately
for i in range(2):
    print(sat_pipe[i])
    fpsat = []
    fnsat = []

    bdfpipe = bdf
    bdfpipe = bdfpipe[bdfpipe.confidence > 0.7]
    bdfpipe_sat = bdfpipe[bdfpipe.detectedBy == sat_pipe[i]]
    print(f'pipeline count: {len(bdfpipe_sat)}')

    bdfweb = bdf_train[bdf_train.detectedBy.str.contains(sat_web[i])]
    bdfweb = bdfweb[(bdfweb.datetime >= min(bdfpipe.datetime)) & (bdfweb.datetime <= max(bdfpipe.datetime))]

    print(f'web count: {len(bdfweb)}')
    for j in range(2):
        if j == 0:  # comparing pipeline to web
            bdfa = bdfpipe_sat
            bdfb = bdfweb
        if j == 1:  # comparing web to pipeline
            bdfa = bdfweb
            bdfb = bdfpipe_sat

        bdfa_latlon = [(lat, lon) for lat, lon in zip(bdfa.latitude, bdfa.longitude)]
        bdfb_latlon = [(lat, lon) for lat, lon in zip(bdfb.latitude, bdfb.longitude)]
        ddeltas = haversine_vector(bdfa_latlon, bdfb_latlon, comb=True)
        print(len(bdfa), len(bdfb))
        unclaimed = np.full(len(bdfb), True)
        for num, (idx, row) in tqdm(enumerate(bdfa.iterrows()), total=len(bdfa)):
            tdeltas = np.abs((row['datetime']-bdfb['datetime']).dt.total_seconds())
            tclose = tdeltas < 5  # 5 seconds away is close enough to be the same probably

            close = tclose*unclaimed  # *dclose
            unclaimed[np.argmax(close)] = False

            if np.sum(close) == 0:
                [fp, fn][j].append(idx)
                [fpsat, fnsat][j].append(idx)
            else:
                both += 1
    print(f'in pipeline, not web: {len(fpsat)}')
    print(f'in web, not pipeline: {len(fnsat)}')

bdf_fp = bdf.loc[fp]
bdf_fp['error'] = r'In $\texttt{machine}$, not in $\texttt{human}$'
bdf_fn = bdf_train.loc[fn]
bdf_fn['error'] = r'In $\texttt{human}$, not in $\texttt{machine}$'
bdf_error = pd.concat([bdf_fp, bdf_fn])
colors = {r'In $\texttt{machine}$, not in $\texttt{human}$': 'brown',
          r'In $\texttt{human}$, not in $\texttt{machine}$': 'darkblue'}
fig, ax = bdf_error.plot_detections(category='error', boundary=['goes-e', 'goes-w'],
                                    s=8, marker='.', figsize=(10, 4), colors=colors, crs=Eck4)
ax.gridlines(color='black', draw_labels=True)
fig.savefig('plots/discrepancies.pdf', bbox_inches='tight')
plt.close()


################################################
# INITIAL LATITUDE HISTOGRAM


def intersecting_polygon(x, step):
    return Polygon([(-180, x-step/2), (-180, x+step/2), (180, x+step/2), (180, x-step/2)])


def intersecting_area(poly):
    return poly.intersection(goes_e).area + poly.intersection(goes_w_i).area/2 + poly.intersection(goes_w_ni).area/2


# get boundary polygons
goes_e, goes_w_i, goes_w_ni = get_boundary(['goes-e', 'goes-w-i', 'goes-w-ni'], crs='epsg:4326')

# correct the West polygons in epsg:4326
coords = goes_w_i.boundary.coords
x = np.array([c[0] for c in coords])
y = np.array([c[1] for c in coords])
x = (x + 50) % 360 - 180
goes_w_i = Polygon([(x[i], y[i]) for i in range(len(x))])
coords = goes_w_ni.boundary.coords
x = np.array([c[0] for c in coords])
y = np.array([c[1] for c in coords])
x = (x + 50) % 360 - 180
goes_w_ni = Polygon([(x[i], y[i]) for i in range(len(x))])

plt.figure(figsize=(4, 3))
step = 2
counts, bins = np.histogram(bdf.latitude, bins=np.arange(-57, 57+step, step=step))

# reference for the error bar calculations used here:
# https://www.pp.rhul.ac.uk/~cowan/atlas/ErrorBars.pdf
lower = counts - chi2.ppf(0.159, df=2*counts)/2
upper = chi2.ppf(1-0.159, df=2*(counts+1))/2 - counts
errors = np.nan_to_num(np.vstack([lower, upper]))

x_shift = bins[:-1] + step/2
midpoint = int(len(counts)/2)

# raw count with error bars
plt.errorbar(x_shift, counts/counts[midpoint], yerr=errors/counts[midpoint],
             linewidth=0, elinewidth=1, label='Raw counts', color='black')

# cosine adjustment with error bars
cos_adjust_factor = np.cos(np.radians(x_shift))
cos_adjust = counts/cos_adjust_factor
plt.errorbar(x_shift+0.60, cos_adjust/cos_adjust[midpoint], yerr=errors/cos_adjust_factor/cos_adjust[midpoint],
             linewidth=0, elinewidth=1, label='Cosine-adjusted', color='grey')

# FOV adjustment with error bars
fov_adjust_factor = np.array([intersecting_area(intersecting_polygon(x, step)) for x in x_shift])
fov_adjust_factor *= np.cos(np.radians(x_shift))
fov_adjust = counts/fov_adjust_factor
plt.errorbar(x_shift-0.60, fov_adjust/fov_adjust[midpoint], yerr=errors/fov_adjust_factor/fov_adjust[midpoint],
             linewidth=0, elinewidth=1, label='FOV-adjusted', color='red')
plt.legend(frameon=False)
plt.xlabel('Latitude [°]')
plt.ylim(0)
plt.ylabel('Rate relative to equator')
plt.savefig('plots/lat-hist.pdf', bbox_inches='tight')

# Plot of the adjustment factors
step = 0.5
counts, bins = np.histogram(bdf.latitude, bins=np.arange(-60, 60+step, step=step))
x_shift = bins[:-1] + step/2
midpoint = int(len(counts)/2)
cos_adjust_factor = np.cos(np.radians(x_shift))
fov_adjust_factor = np.array([intersecting_area(intersecting_polygon(x, step)) for x in x_shift])
fov_adjust_factor *= np.cos(np.radians(x_shift))
plt.figure(figsize=(4, 3))
plt.plot(x_shift, np.ones(len(x_shift)), color='black', label='Constant area')
plt.plot(x_shift, cos_adjust_factor, color='grey', label='Cosine adjustment')
plt.plot(x_shift, fov_adjust_factor/fov_adjust_factor[midpoint], color='red', label='FOV adjustment')
plt.legend(frameon=False)
plt.ylim(0)
plt.xlabel('Latitude [°]')
plt.ylabel('Normalized area at latitude')
plt.savefig('plots/lat-area.pdf', bbox_inches='tight')

#################################
# SOLAR HOUR PLOTS
bdf = BolideDataFrame(source='csv', files='data/pipeline-dedup.csv', annotate=False)
bdf = bdf[bdf.confidence > 0.7]

plt.figure(figsize=(4, 3))
plt.hist(bdf.solarhour, color='black', edgecolor='white', linewidth=0.5, bins=24, range=(0, 24))
plt.xticks([0, 6, 12, 18, 24])
plt.ylabel('Count')
plt.xlabel('Solar hour')
plt.axvline(x=6, color='red', linestyle='--')
plt.title('GLM bolide detections by solar hour')
plt.xlim(0, 24)
plt.savefig('plots/solarhour.pdf', bbox_inches='tight')

bdf['sollon'] = [c.ra.deg for c in get_sun(Time(bdf.datetime))]

subsets = [bdf.sollon.between(45, 135),
           bdf.sollon.between(135, 225),
           bdf.sollon.between(225, 315),
           ~bdf.sollon.between(45, 315)]
seasons = ['Summer', 'Fall', 'Winter', 'Spring']
colors = ['darkgreen', 'orangered', 'lightblue', 'lightpink']
plt.figure(figsize=(4, 3))
for i in range(4):
    subset = subsets[i]
    season = seasons[i]
    plt.hist(bdf[subset].solarhour, histtype='step', bins=24, range=(0, 24), label=season)
plt.xticks([0, 6, 12, 18, 24])
plt.xlim(0, 24)
plt.ylabel('Count')
plt.xlabel('Solar hour')
plt.legend(frameon=False)
plt.savefig('plots/solarhour-seasons.pdf', bbox_inches='tight')

########################################
# SOLAR LONGITUDE PLOT
fig, ax = plt.subplots(1, 1, figsize=(8, 3))
plt.hist(bdf['sollon'], bins=360, color='black')
plt.xlim(0, 360)
plt.axvline(235, linestyle=':', label='Leonids', color='grey')
ax.text(236, 130, 'Leonids\n70km/s')
plt.axvline(140, linestyle=':', label='Perseids', color='grey')
ax.text(142, 50, 'Perseids\n 59km/s')

plt.axvline(128, linestyle=':', label='Southern Delta Aquariids', color='grey')
ax.text(128, 25, 'Southern\n Delta Aquariids\n 41km/s', horizontalalignment='right')

plt.axvline(209, linestyle=':', label='Orionids', color='grey')
ax.text(208, 30, 'Orionids\n 66km/s', horizontalalignment='right')
plt.axvline(46, linestyle=':', label=r'$\eta$ Aquariids', color='grey')
ax.text(48, 50, r'$\eta$ Aquariids' + '\n 66km/s')

plt.axvline(32, linestyle=':', label='April Lyrids', color='grey')
ax.text(30, 25, 'April\n Lyrids\n 47km/s', horizontalalignment='right')

plt.axvline(262, linestyle=':', label='Geminids', color='grey')
ax.text(263, 25, 'Geminids\n 34km/s')

plt.axvline(283, linestyle=':', label='Quadrantids', color='grey')
ax.text(283, 50, 'Quadrantids\n 41km/s')
plt.xlabel('Solar longitude (°)')
plt.ylabel('Count')
plt.savefig('plots/sollon.pdf', bbox_inches='tight')

###########################################
# USG (NON)UNIFORMITY
bdf = BolideDataFrame(source='usg')
fig, ax = bdf.plot_density(figsize=(10, 4), bandwidth=30, crs=Eck4)
ax.gridlines(color='black', draw_labels=True)
fig.savefig('plots/usg-density.pdf', dpi=300, bbox_inches='tight')

kde = KernelDensity(kernel="gaussian", bandwidth=20)
data = np.array(bdf.longitude[pd.isna(bdf.longitude) == False])
wrapped = data
wrapped = np.append(wrapped, data+360)
wrapped = np.append(wrapped, data-360)
kde.fit(wrapped[:, np.newaxis])
x_plot = np.linspace(-180, 180, 200)
density = np.exp(kde.score_samples(x_plot[:, np.newaxis]))
usg_discrep = max(density)/min(density)

discrep = []
for i in tqdm(range(10000)):
    data = np.random.random(len(bdf.longitude))*360-180
    wrapped = data
    wrapped = np.append(wrapped, data+360)
    wrapped = np.append(wrapped, data-360)
    kde = KernelDensity(kernel="gaussian", bandwidth=30)
    kde.fit(wrapped[:, np.newaxis])
    x_plot = np.linspace(-55, 55, 200)
    density = np.exp(kde.score_samples(x_plot[:, np.newaxis]))
    discrep.append(max(density)/min(density))

plt.figure(figsize=(4, 3))
plt.hist(discrep, bins=100, density=True, histtype='step', linewidth=2, color='black', label="Simulated")
plt.axvline(usg_discrep, color='red', linewidth=2, label="USG")
plt.xlabel('Maximum of KDE / Minimum of KDE')
plt.ylabel('Density')
plt.legend(frameon=False)
plt.savefig('plots/usg-lon.pdf', bbox_inches='tight')

with open('models/reg.pkl', 'rb') as f:
    result = pd.read_pickle(f)
data = result['data']
laea = ccrs.LambertAzimuthalEqualArea(central_longitude=GOES_E_LON, central_latitude=0.0)
g16 = data[data.sat == 'g16']
density = g16['count']/(g16['area']*510072000)
plot_polygons(g16, data=density, crs=laea, label='Bolide density [km$^{-2}$]', show=False,
              filename='bolide-density-g16', second='side', figsize=(10, 4),
              extent=[-140, -10.4, -60, 60])

###################################
# AOI SENSITIVITY DATA PLOT
if os.path.exists('data/glm-bandpass.txt'):
    fov_truth = pd.read_csv('data/glm-bandpass.txt', skiprows=9, sep='\t')
    plt.figure(figsize=(4, 3))
    plt.plot(fov_truth.AOI, fov_truth.Background, color='black', label="Integrated transmittance\n over filter passband")
    plt.plot(fov_truth.AOI, fov_truth.Lightning, color='gray', label="Fraction of lightning\n signal reaching detector")
    plt.xlabel('Angle of incidence [°]')
    plt.ylabel('Measured GLM filter transmittance')
    plt.axvline(8.34, linestyle=':', color='black', label='Largest AOI in FOV')
    plt.legend(frameon=False)
    plt.savefig('plots/sensor-aoi.pdf', bbox_inches='tight')

###########################
# ECLON-ECLAT DENSITY

data = np.loadtxt('fortran/pvil.dat', skiprows=0).reshape((360, 180, 100))
d = np.sum(data, axis=2).T
d = d / np.cos(np.radians(np.linspace(-89.5, 89.5, 180)))[:, np.newaxis]

fig, ax = plt.subplots(figsize=(8, 3))
for loc in ['bottom', 'top', 'left', 'right']:
    ax.spines[loc].set_color('white')
img = ax.imshow(d/np.sum(d), cmap='Greys_r', extent=[90, -270, -90, 90])
ax.set_xticks([90, 60, 30, 0, -30, -60, -90, -120, -150, -180, -210, -240, -270])
plt.yticks([-90, -60, -30, 0, 30, 60, 90])
ticks = np.array(list(ax.get_xticks()))
ticks[ticks < 0] += 360
ax.set_xticklabels([int(tick) for tick in ticks])
plt.colorbar(img, label='Density')
ax.tick_params(axis='both', color='white')

plt.xlabel('Sun-centered ecliptic longitude [°]')
plt.ylabel('Ecliptic latitude [°]')

plt.savefig('plots/eclon-eclat.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/eclon-eclat.pdf', dpi=300, bbox_inches='tight')

############################
# SOLAR PLOTS
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
ax1 = axs[0]
ax2 = axs[1]

bdf_glm = BolideDataFrame(source='csv', files='data/pipeline-dedup.csv', annotate=True)
bdf_glm = bdf_glm[bdf_glm.confidence > 0.7]
bdf_usg = BolideDataFrame(source='usg', annotate=True)

theory_exists = os.path.exists('impacts/impact-dists.csv')
if theory_exists:
    df = pd.read_csv('impacts/impact-dists.csv')

# SOLAR HOUR
bdfs = [bdf_glm, bdf_usg]
colors = ['black', 'grey']
labels = ['GLM data', 'USG data']
shifts = [0, 0]
steps = [1, 2]
for i in range(2):
    bdf = bdfs[i]
    color = colors[i]
    shift = shifts[i]
    step = steps[i]
    counts, bins = np.histogram(np.abs(bdf.solarhour), bins=np.arange(0, 24+step, step=step), density=False)
    dens = counts/(np.sum(counts)*step)
    # https://www.pp.rhul.ac.uk/~cowan/atlas/ErrorBars.pdf
    lower = counts - chi2.ppf(0.05, df=2*counts)/2
    upper = chi2.ppf(1-0.05, df=2*(counts+1))/2 - counts
    errors = np.nan_to_num(np.vstack([lower, upper]))
    x_shift = bins[:-1] + step/2
    ax1.errorbar(x_shift+shift, dens, yerr=errors/(np.sum(counts)*step), linewidth=0, elinewidth=1, color=color)
if theory_exists:
    ax1.plot(df.x_solarhour, df.v0_solarhour,
             label=r'$V_{\infty}\geq0$ km/s, whole Earth', linestyle='--', color='blue')
    ax1.plot(df.x_solarhour, df.v50_solarhour,
             label=r'$V_{\infty}\geq50$ km/s, whole Earth', linestyle='--', color='red')
    ax1.plot(df.x_solarhour, df.v0_solarhour_glm,
             label=r'$V_{\infty}\geq0$ km/s, $\in$ GLM FOV', linestyle=':', color='blue')
    ax1.plot(df.x_solarhour, df.v50_solarhour_glm,
             label=r'$V_{\infty}\geq50$ km/s, $\in$ GLM FOV', linestyle=':', color='red')
    ax1.set_xlim(0, 24)
step = steps[0]
ax1.hist(bdf_glm.solarhour, density=True, bins=np.arange(0, 24+step, step=step),
         histtype='step', color='black', label=r'GLM, $n$='+str(len(bdf_glm)))
step = steps[1]
ax1.hist(bdf_usg.solarhour, density=True, bins=np.arange(0, 24+step, step=step),
         histtype='step', color='grey', label=r'USG, $n$='+str(len(bdf_usg)))
ax1.set_ylabel('Density')
ax1.set_xlabel('Solar hour')
ax1.set_xticks([0, 6, 12, 18, 24])
ax1.legend(frameon=False)

# SOLAR ALTITUDE
bdfs = [bdf_glm, bdf_usg]
colors = ['black', 'grey']
labels = ['GLM data', 'USG data']
shifts = [0, 0]
steps = [7.5, 15]
for i in range(2):
    bdf = bdfs[i]
    color = colors[i]
    shift = shifts[i]
    step = steps[i]
    counts, bins = np.histogram(bdf.sun_alt_app, bins=np.arange(-90, 90+step, step=step), density=False)
    dens = counts/(np.sum(counts)*step)
    # https://www.pp.rhul.ac.uk/~cowan/atlas/ErrorBars.pdf
    lower = counts - chi2.ppf(0.05, df=2*counts)/2
    upper = chi2.ppf(1-0.05, df=2*(counts+1))/2 - counts
    errors = np.nan_to_num(np.vstack([lower, upper]))
    x_shift = bins[:-1] + step/2
    ax2.errorbar(x_shift+shift, dens, yerr=errors/(np.sum(counts)*step), linewidth=0, elinewidth=1, color=color)

if theory_exists:
    ax2.plot(df.x_sun_alt, df.v0_sun_alt,
             label=r'Predicted, $V_{\infty}\geq0$ km/s, whole Earth', linestyle='--', color='blue')
    ax2.plot(df.x_sun_alt, df.v50_sun_alt,
             label=r'Predicted, $V_{\infty}\geq50$ km/s, whole Earth', linestyle='--', color='red')
    ax2.plot(df.x_sun_alt, df.v0_sun_alt_glm,
             label=r'Predicted, $V_{\infty}\geq0$ km/s, $\in$ GLM FOV', linestyle=':', color='blue')
    ax2.plot(df.x_sun_alt, df.v50_sun_alt_glm,
             label=r'Predicted, $V_{\infty}\geq50$ km/s, $\in$ GLM FOV', linestyle=':', color='red')

ax2.set_xlim(-90, 90)
ax2.set_xticks([-90, -60, -30, 0, 30, 60, 90])
step = steps[0]
ax2.hist(bdf_glm.sun_alt_app, density=True, bins=np.arange(-90, 90+step, step=step),
         histtype='step', color='black', label='GLM data')
step = steps[1]
ax2.hist(bdf_usg.sun_alt_app, density=True, bins=np.arange(-90, 90+step, step=step),
         histtype='step', color='grey', label='USG data')
ax2.set_xlabel('Apparent solar altitude [°]')

handles, labels = axs[0].get_legend_handles_labels()

axs[0].get_legend().remove()

left = -1.1
yloc = 1.0
if theory_exists:
    l1 = plt.legend(handles[:4], labels[:4], loc=(left, yloc), frameon=False, ncols=2,
                    handletextpad=0.3, columnspacing=1, title=r'$\textbf{Predicted}$')
    plt.legend(handles[4:], labels[4:], loc=(left+1.55, yloc), frameon=False, ncols=1,
               handletextpad=0.3, columnspacing=1, title=r'$\textbf{Observed}$')
    plt.gca().add_artist(l1)
else:
    plt.legend(handles, labels, loc=(left+1.55, yloc), frameon=False, ncols=1,
               handletextpad=0.3, columnspacing=1, title=r'$\textbf{Observed}$')

plt.savefig('plots/solar-theory.png', dpi=300, bbox_inches='tight')
plt.savefig('plots/solar-theory.pdf', bbox_inches='tight')
