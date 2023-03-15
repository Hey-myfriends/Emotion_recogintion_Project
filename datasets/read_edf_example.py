import mne
import numpy as np
import json, os
import pandas as pd
import matplotlib.pyplot as plt

all_edf = [p for p in os.listdir('../高负荷生理采集实验-1023/04 EGG数据/') if p.endswith('.edf')]
print(len(all_edf), all_edf)
os.makedirs('./fig', exist_ok=True)

# path = '../高负荷生理采集实验-1023/04 EGG数据/20221023131626_CLF_1.edf'
breakpoint()
useful = 0
for p in all_edf:
    print("check {}...".format(p))
    path = os.path.join('../高负荷生理采集实验-1023/04 EGG数据/', p)
    os.makedirs('./fig/{}'.format(p[:-4]), exist_ok=True)
    raw_data = mne.io.read_raw_edf(path, preload=True)
    eeg = raw_data.to_data_frame()
    # breakpoint()
    eeg_np = eeg.to_numpy()
    for i in range(eeg_np.shape[1]):
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))
        ax.plot(eeg_np[:, i])
        ax.set(title=eeg.columns[i])
        fig.savefig('./fig/{}/{}.png'.format(p[:-4], eeg.columns[i]), dpi=300, bbox_inches='tight')
        plt.close()
    # breakpoint()
    # print(type(eeg))
    useful += 1
print("useful = {}".format(useful))