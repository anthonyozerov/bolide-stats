import pickle
from plotting import plot_lat_result, plot_fov_result, plot_polygons
import pandas as pd
import sys

model_name = sys.argv[1]

def load_model(name):
    with open(name+'.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

fov_truth = pd.read_csv('data/glm-bandpass.txt', skiprows=9, sep='\t')
if 'eclip' in model_name:
    leo_truth = pd.read_csv('impacts/leonids-eclip-dist.csv')
    lat_truth = pd.read_csv('impacts/high-vel-eclip-dist.csv')
    spacer='ecliptic '
else:
    leo_truth = pd.read_csv('impacts/leonids-dist.csv')
    lat_truth = pd.read_csv('impacts/high-vel-dist.csv')
    spacer=''
print(leo_truth)
print(lat_truth)
#print(fov_truth)

model = load_model(f'models/{model_name}')
plot_fov_result(model, 'Bolide rate dependent on distance from nadir', f'{model_name}-fov', normalize=False, angle=True, truth=fov_truth)
plot_lat_result(model, f'Bolide rate dependent on {spacer}latitude', f'{model_name}', normalize=True, symmetric=False, theory=lat_truth)
plot_lat_result(model, f'Leonid bolide rate dependent on {spacer}latitude', f'{model_name}-leo', shower='LEO', normalize=True, symmetric=False, theory=leo_truth)
plot_lat_result(model, f'Leonid bolide rate dependent on {spacer}latitude', f'{model_name}-leo-sym', shower='LEO', normalize=True, symmetric=True, theory=leo_truth)
plot_lat_result(model, f'Bolide rate dependent on {spacer}latitude', f'{model_name}-sym', normalize=True, symmetric=True, theory=lat_truth)
exit()
plot_lat_result(model, f'Bolide rate dependent on {spacer}latitude', f'{model_name}-unnorm', normalize=False, symmetric=False)
#plot_polygons(model['data'][model['data']['LEO']==0])
print(model['data']['duration'])
