# This script de-duplicates a csv of GLM pipeline data.
# It doesn't combine stereo detections, but does get rid of
# repeated detections arising from an event split across multiple
# NetCDF4 files. It does this by simply throwing away detections within
# 5s of an existing event, which due to the low total number of events will
# work well. This script also filters for a confidence > 0.25

from bolides import BolideDataFrame
from tqdm import tqdm
import numpy as np

bdf = BolideDataFrame(source='csv', files='data/pipeline.csv', annotate=False)
bdf = bdf.sort_values(by=['confidence'], ascending=False)
bdf = bdf[bdf.confidence > 0.25]

idx_g16 = []
bdf_subset = bdf[bdf.detectedBy == 'G16']
for idx, row in tqdm(bdf_subset.iterrows()):
    dts = bdf_subset.datetime[idx_g16]
    if len(idx_g16) == 0 or np.min(np.abs(dts-row.datetime).dt.total_seconds()) > 5:
        idx_g16.append(idx)

idx_g17 = []
bdf_subset = bdf[bdf.detectedBy == 'G17']
for idx, row in tqdm(bdf_subset.iterrows()):
    dts = bdf_subset.datetime[idx_g17]
    if len(idx_g17) == 0 or np.min(np.abs(dts-row.datetime).dt.total_seconds()) > 5:
        idx_g17.append(idx)
