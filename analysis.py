import pickle
from plotting import plot_lat_result, plot_fov_result, plot_polygons
import pandas as pd
import sys
import cartopy.crs as ccrs
from bolides.constants import GLM_STEREO_MIDPOINT, GOES_E_LON

if not os.path.exists('plots'):
    os.makedirs('plots')

goes_e_eck4 = ccrs.EckertIV(central_longitude=GOES_E_LON)

model_name = sys.argv[1]
velplot = '0.' in model_name

def load_model(name):
    with open(name+'.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

fov_truth = pd.read_csv('data/glm-bandpass.txt', skiprows=9, sep='\t')
#if 'eclip' in model_name:
#    leo_truth = pd.read_csv('impacts/leonids-eclip-dist.csv')
#    lat_truth = pd.read_csv('impacts/high-vel-eclip-dist.csv')
#    spacer='ecliptic '
#else:
leo_truth = pd.read_csv('impacts/leonids-dist.csv')
lat_truth = pd.read_csv('impacts/high-vel-dist.csv')
print(leo_truth)
print(lat_truth)
#print(fov_truth)

model = load_model(f'models/{model_name}')

if not velplot:
    # plot polygons
    model['data']['area'] *= 510064472 # area of Earth
    print(model['data'])
    if 'LEO' in model['data'].columns:
        model['data'] = model['data'][model['data']['LEO']==0]
    polygon_data = model['data'][model['data']['sat']=='g16']
    #plot_polygons(polygon_data, crs=goes_e_eck4, column='area', label='Polygon area (km$^{2}$)', filename=f'{model_name}-g16')
    #plot_polygons(polygon_data, crs=goes_e_eck4, column='density', label='Bolide density (km$^{-2}$)', filename=f'{model_name}-g16')
    polygon_data = model['data'][model['data']['sat']=='g17']
    #plot_polygons(polygon_data, column='duration', label='Normalized GLM observation time', filename=f'{model_name}-g17')

# plot posteriors
results = model['results']
for num, result in enumerate(results):
    if len(results) > 1:
        sat = ['GOES-16', 'GOES-17'][num]
        title_suffix = f' ({sat})'
        file_suffix = ['-g16', '-g17'][num]
    else:
        title_suffix=''
        file_suffix=''
    if velplot:
        vel = 'Confidence $\geq$ ' + model_name.split('-leo-')[1]
    title = vel if velplot else f'Bolide rate dependent on distance from nadir{title_suffix}'
    plot_fov_result(result, title, f'{model_name}-fov{file_suffix}', normalize=True, angle=True, truth=fov_truth)
    title = vel if velplot else f'Bolide rate dependent on latitude{title_suffix}'
    plot_lat_result(result, title, f'{model_name}{file_suffix}', normalize=True, symmetric=False, theory=lat_truth)
    title = vel if velplot else f'Bolide rate dependent on latitude{title_suffix}'
    plot_lat_result(result, title, f'{model_name}-sym{file_suffix}', normalize=True, symmetric=True, theory=lat_truth)
    title = vel if velplot else f'Bolide rate dependent on latitude{title_suffix}'
    plot_lat_result(result, title, f'{model_name}-unnorm{file_suffix}', normalize=False, symmetric=False)
    if 'LEO' in model['data'].columns:
        plot_lat_result(result, f'Leonid bolide rate dependent on latitude{title_suffix}', f'{model_name}-leo{file_suffix}', shower='LEO', normalize=True, symmetric=False, theory=leo_truth)
        plot_lat_result(result, f'Leonid bolide rate dependent on latitude{title_suffix}', f'{model_name}-leo-sym{file_suffix}', shower='LEO', normalize=True, symmetric=True, theory=leo_truth)
