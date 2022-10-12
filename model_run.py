from bolides import BolideDataFrame
from model import full_model
import pickle

def write_result(result, name):
    with open('models/'+name+'.pkl', 'wb') as f:
        pickle.dump(result, f)

bdf = BolideDataFrame(source='csv', files='data/pipeline.csv', annotate=False)
bdf_subset = bdf[bdf.confidence >= 0.7]
g16 = bdf_subset[bdf_subset.detectedBy == 'G16']
g17 = bdf_subset[bdf_subset.detectedBy == 'G17']

param_leo = full_model(g16, g17, nonparam=False, n_points=100, showers=['LEO'])
write_result(param_leo, 'param_leo.pkl')
