import pickle
from plotting import plot_lat_result, plot_fov_result, plot_polygons


def load_model(name):
    with open(name+'.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


model = load_model('models/param-leo')
# plot_lat_result(model, 'Bolide rate dependent on latitude', 'param-leo', normalize=True, symmetric=True)
plot_polygons(model['data'][model['data']['LEO']==0])
print(model['data']['duration'])
