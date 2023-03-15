from datasets.config import config
import os
import numpy as np
from tqdm import tqdm

rootpth = config['dataset_out_dir']
sam_all = os.listdir(rootpth)

shp = None
shape_all = {}
breakpoint()
for sam in tqdm(sam_all):
    sample = np.loadtxt(os.path.join(rootpth, sam))
    if shp is None:
        shp = sample.shape
        shape_all[shp] = 1
    else:
        if shp != sample.shape:
            print(shp, sam)
            shp = sample.shape
            shape_all[shp] = 1
        else:
            shape_all[shp] += 1
breakpoint()
print(shape_all)