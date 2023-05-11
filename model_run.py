from bolides import BolideDataFrame
from model import full_model
import pickle
import os

if not os.path.exists('models'):
    os.makedirs('models')


def write_result(result, name):
    with open('models/'+name+'.pkl', 'wb') as f:
        pickle.dump(result, f)


def run_model(name, **kwargs):
    if name+'.pkl' not in os.listdir('models'):
        print(f'running {name}')
        result = full_model(**kwargs)
        write_result(result, name)
        print('---------------')


print('loading data')
bdf = BolideDataFrame(source='csv', files='data/pipeline.csv', annotate=False)
bdf_subset = bdf[bdf.confidence >= 0.7]
g16 = bdf_subset[bdf_subset.detectedBy == 'G16']
g17 = bdf_subset[bdf_subset.detectedBy == 'G17']

#run_model('param-leo-jax', g16=g16, g17=g17, nonparam=False, n_points=1000, showers=['LEO'])
#run_model('param-separate', g16=g16, g17=g17, nonparam=False, n_points=1000, showers=[], separate=True)
#run_model('nonparam', g16=g16, g17=g17, nonparam=True, n_points=500, showers=[])
run_model('l1', g16=g16, g17=g17, nonparam=False, n_points=1000, f_lat='x^1', f_fov='', showers=[])
run_model('f1', g16=g16, g17=g17, nonparam=False, n_points=1000, f_lat='', f_fov='x^1', showers=[])
run_model('l1f1', g16=g16, g17=g17, nonparam=False, n_points=1000, f_lat='x^1', f_fov='x^1', showers=[])
run_model('l12f1', g16=g16, g17=g17, nonparam=False, n_points=1000, f_lat='x^1+x^2', f_fov='x^1', showers=[])
run_model('l1f12', g16=g16, g17=g17, nonparam=False, n_points=1000, f_lat='x^1', f_fov='x^1+x^2', showers=[])
run_model('l12f12', g16=g16, g17=g17, nonparam=False, n_points=1000, f_lat='x^1+x^2', f_fov='x^1+x^2', showers=[])
run_model('l123f12', g16=g16, g17=g17, nonparam=False, n_points=1000, f_lat='x^1+x^2+x^3', f_fov='x^1+x^2', showers=[])
run_model('l12f123', g16=g16, g17=g17, nonparam=False, n_points=1000, f_lat='x^1+x^2', f_fov='x^1+x^2+x^3', showers=[])
run_model('l124f12', g16=g16, g17=g17, nonparam=False, n_points=1000, f_lat='x^1+x^2+x^4', f_fov='x^1+x^2', showers=[])
run_model('l12f12L', g16=g16, g17=g17, nonparam=False, n_points=1000, f_lat='x^1+x^2', f_fov='x^1+x^2',biases=['land'], showers=[])
run_model('l12f12F', g16=g16, g17=g17, nonparam=False, n_points=1000, f_lat='x^1+x^2', f_fov='x^1+x^2',biases=['flash_dens'], showers=[])

for conf in [0.25, 0.5, 0.7, 0.8, 0.9]:
    bdf_subset = bdf[bdf.confidence >= conf]
    g16 = bdf_subset[bdf_subset.detectedBy == 'G16']
    g17 = bdf_subset[bdf_subset.detectedBy == 'G17']
    run_model(f'param-leo-{conf}', g16=g16, g17=g17, nonparam=False, n_points=1000, showers=['LEO'])

# run_model('param-leo-ecliptic', g16=g16, g17=g17, nonparam=False, n_points=1000, showers=['LEO'], ecliptic=True)
# run_model('nonparam', g16=g16, g17=g17, nonparam=True, n_points=100, showers=[])
