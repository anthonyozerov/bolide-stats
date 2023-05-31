from bolides import BolideDataFrame
from model import full_model
import pickle
import os
from pathlib import Path

shower_data = {'LEO': 'impacts/leonids-dist.csv', 'PER': 'impacts/perseids-dist.csv'}

if not os.path.exists('models'):
    os.makedirs('models')

def write_result(result, name):
    with open('models/'+name+'.pkl', 'wb') as f:
        pickle.dump(result, f)


def run_model(name, **kwargs):
    if name+'.pkl' not in os.listdir('models'):
        # write so that another instance of model_run doesn't do the same work
        Path(f'models/{name}.pkl').touch()
        print(f'running {name}')
        result = full_model(**kwargs)
        write_result(result, name)
        print('---------------')


print('loading data')
bdf = BolideDataFrame(source='csv', files='data/pipeline.csv', annotate=False)
bdf_subset = bdf[bdf.confidence >= 0.7]
g16 = bdf_subset[bdf_subset.detectedBy == 'G16']
g17 = bdf_subset[bdf_subset.detectedBy == 'G17']

f = 'x^1+x^2+x^3+x^4+x^5+x^6'

run_model('reg', g16=g16, g17=g17, n_points=1000, f_lat=f, f_fov=f, showers=[],biases=['flash_dens','land'])
run_model('reg-separate', g16=g16, g17=g17, n_points=1000, f_lat=f, f_fov=f, showers=[],biases=['flash_dens','land'], separate=True)
run_model('reg-S', g16=g16, g17=g17, n_points=1000, f_lat=f, f_fov=f, showers=[],biases=['flash_dens','land','stereo'])
run_model('leo', g16=g16, g17=g17, n_points=1000, f_lat=f, f_fov=f, showers=['LEO'],biases=['flash_dens','land'])
run_model('leo-known', g16=g16, g17=g17, n_points=1000, f_lat=f, f_fov=f, showers=['LEO'],biases=['flash_dens','land'], shower_data=shower_data)
run_model('leoper', g16=g16, g17=g17, n_points=1000, f_lat=f, f_fov=f, showers=['LEO','PER'],biases=['flash_dens','land'])
run_model('leoper-known', g16=g16, g17=g17, n_points=1000, f_lat=f, f_fov=f, showers=['LEO','PER'],biases=['flash_dens','land'], shower_data=shower_data)

for conf in [0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
    bdf_subset = bdf[bdf.confidence >= conf]
    g16 = bdf_subset[bdf_subset.detectedBy == 'G16']
    g17 = bdf_subset[bdf_subset.detectedBy == 'G17']
    run_model(f'reg-{conf}', g16=g16, g17=g17, n_points=1000, f_lat=f, f_fov=f, biases=['flash_dens','land'])

print('loading web data')
bdf_web = BolideDataFrame(source='csv', files='data/web.csv', annotate=False)
bdf_web = bdf_web[(bdf_web.datetime>min(bdf.datetime)) & (bdf_web.datetime<max(bdf.datetime))]
g16 = bdf_web[bdf_web.detectedBy.str.contains('GLM-16')]
g17 = bdf_web[bdf_web.detectedBy.str.contains('GLM-17')]
run_model('human-reg-S', g16=g16, g17=g17, n_points=1000, f_lat=f, f_fov=f, showers=[], biases=['flash_dens','land','stereo'])
