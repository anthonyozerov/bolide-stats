import pickle
from plotting import plot_lat_result, plot_fov_result, plot_polygons
import pandas as pd


def load_model(name):
    with open(name+'.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

fov_truth = pd.read_csv('data/glm-bandpass.txt', skiprows=9, sep='\t')
#print(fov_truth)

model = load_model('models/param-leo')
plot_fov_result(model, 'Bolide rate dependent on latitude', 'param-leo-fov', normalize=False, angle=True, truth=fov_truth)
plot_lat_result(model, 'Bolide rate dependent on latitude', 'param-leo', normalize=True, symmetric=False)
plot_lat_result(model, 'Bolide rate dependent on latitude', 'param-leo-leo', shower='LEO', normalize=True, symmetric=False)
plot_lat_result(model, 'Bolide rate dependent on latitude', 'param-leo-sym', normalize=True, symmetric=True)
plot_lat_result(model, 'Bolide rate dependent on latitude', 'param-leo-unnorm', normalize=False, symmetric=False)
#plot_polygons(model['data'][model['data']['LEO']==0])
print(model['data']['duration'])
