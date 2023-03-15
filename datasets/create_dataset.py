
# sample saving format: num_seg_subseg_label.txt
import numpy as np
import pandas as pd
import os, sys
import mne
from tqdm import tqdm

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
    print(sys.path[0])
add_path('/home/bebin.huang/Code/emotion_recognition/code')
from get_logger import get_root_logger
from datasets.config import config
LOGGER = get_root_logger(console_out=True, logName="create_dataset.log")

def read_edf_file(edf_path):
    raw_data = mne.io.read_raw_edf(edf_path, preload=False)
    eeg = raw_data.to_data_frame() # pd.DataFrames
    return eeg

def extract_seg(cur_eeg_all: np.array, start_shift, length, fs):
    # breakpoint()
    if isinstance(start_shift, int):
        assert isinstance(length, int)
        return cur_eeg_all[start_shift*fs:(start_shift+length)*fs]
    
    assert isinstance(start_shift, list) and isinstance(length, list)
    segs = []
    for s, l in zip(start_shift, length):
        segs.append(cur_eeg_all[s*fs:(s+l)*fs])
    return segs

def split_to_samples(data_all):
    # breakpoint()
    seg_to_samples_dict = config.get('seg_to_samples', None)
    if seg_to_samples_dict is None:
        LOGGER.error("Error! Please provide segment config.")
        exit()
    os.makedirs(config['dataset_out_dir'], exist_ok=True)
    # breakpoint()
    fs = config['fs']['eeg']
    window_length = seg_to_samples_dict['window_length'] * fs
    stepsize = seg_to_samples_dict['stepsize'] * fs
    total, total0, total1 = 0, 0, 0
    for i in tqdm(range(len(data_all['segments']))):
        # breakpoint()
        cur_seg = data_all['segments'][i]
        cur_label = data_all['seg_labels'][i]
        cur_info = data_all['info'][i]
        if not isinstance(cur_seg, list):
            cur_seg = [cur_seg]

        for j in range(len(cur_seg)):
            assert cur_seg[j].shape[0] > window_length, f'cur_seg[j].shape[0] = {cur_seg[j].shape[0]}, window_length = {window_length}, info = {cur_info}'
            end_point = window_length
            # breakpoint()
            while end_point <= cur_seg[j].shape[0]:
                # if end_point > cur_seg[j].shape[0]:
                    # end_point = cur_seg[j].shape[0]
                sample = cur_seg[j][end_point-window_length:end_point, :]
                end_point += stepsize
                np.savetxt(os.path.join(config['dataset_out_dir'], str(total).zfill(5) + "_{}_{}_{}.txt".format(i, j, cur_label)), sample) # save format: num_seg_subseg_label.txt
                total += 1
                if cur_label: total1 += 1
                else: total0 += 1
    LOGGER.info("Produce a total of {} samples, {} positive and {} negative.".format(total, total1, total0))


def create_dataset():
    # breakpoint()
    for key, value in config.items():
        LOGGER.info("{}: {}".format(key, str(value)))
    rootpath = config['rootpath']
    fs = config['fs']['eeg']
    eeg_start_time = config['eeg_start_time']#[:2]
    eeg_start_shift = config['eeg_start_shift']#[:2]
    length = config['length']#[:2]
    seg_labels = config['seg_labels']#[:2]
    all_edf_dict = {eval(p.split('_')[0][-6:])+3: p for p in os.listdir(rootpath) if p.endswith('.edf')}
    LOGGER.info("edf details: {}".format(str(all_edf_dict)))

    # breakpoint()
    pop_keys = ('time', 'X', 'Y', 'Z')
    data_all = dict(segments = [], seg_labels = [], channels = None, info = [])
    for i, start_time in enumerate(eeg_start_time):
        assert start_time in all_edf_dict
        edf_path = all_edf_dict[start_time]
        LOGGER.info("Extracting data from {}".format(edf_path))
        cur_eeg_all = read_edf_file(os.path.join(rootpath, edf_path))
        for k in pop_keys: 
            if k in cur_eeg_all: cur_eeg_all.pop(k)
        candidate_seg = extract_seg(cur_eeg_all.to_numpy(), eeg_start_shift[i], length[i], fs)

        if data_all['channels'] is None:
            data_all['channels'] = list(cur_eeg_all.columns)
            LOGGER.info("channels: {}".format(str(data_all['channels'])))
        data_all['segments'].append(candidate_seg)
        data_all['seg_labels'].append(seg_labels[i])
        data_all['info'].append(edf_path)
    # breakpoint()
    data_all_details = []
    for i in range(len(data_all['segments'])):
        cur_seg = data_all['segments'][i]
        if isinstance(cur_seg, list):
            for s in cur_seg:
                data_all_details.append([s.shape[0]//fs, data_all['seg_labels'][i]])     
            continue   
        data_all_details.append([cur_seg.shape[0]//fs, data_all['seg_labels'][i]])
    LOGGER.info("seg details: {}".format(str(data_all_details)))
    len0, len1 = 0, 0
    for l in data_all_details:
        if l[1] == 0: len0 += l[0]
        elif l[1] == 1: len1 += l[0]
    
    LOGGER.info("label 0:1 = {}:{} sec. ratio = {:.2f}".format(len0, len1, len0/len1 if len1 else -1))
    split_to_samples(data_all)
    LOGGER.info("Create dataset successfully.")

if __name__ == "__main__":
    create_dataset()
