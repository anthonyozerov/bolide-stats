import os
import numpy as np

D = np.zeros((360,720))

# iterate over files in the clouds subdirectory,
# loading the csv files and adding them to the running sum D
for fname in os.listdir('clouds'):
    d = np.loadtxt(f'clouds/{fname}', delimiter=',')
    d[d>1] = np.nan
    D += d
# save the averaged cloud data
np.savetxt(f'cloud-avg.txt', D/len(os.listdir('clouds')))

# make and save a nice picture
import matplotlib.pyplot as plt
plt.imshow(D)
plt.savefig('cloud-avg.png')
