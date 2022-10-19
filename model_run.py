from bolides import BolideDataFrame
from model import full_model
import pickle
import os


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

run_model('param-leo', g16=g16, g17=g17, nonparam=False, n_points=100, showers=['LEO'])
run_model('nonparam', g16=g16, g17=g17, nonparam=True, n_points=100, showers=[])
