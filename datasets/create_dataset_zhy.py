# data format: num_subject_seg_label.txt
import numpy as np
import pandas as pd
import os, sys
import mne
from tqdm import tqdm
import datetime, pytz

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
    print(sys.path[0])
add_path('/home/bebin.huang/Code/emotion_recognition/Emotion-Recognition-with-4DRCNN-and-DGCNN_LSTM')
from get_logger import get_root_logger
from datasets.config_zhy import config
suffix = config.get('suffix', '')
LOGGER = get_root_logger(console_out=True, logName="create_dataset_{}.log".format(suffix))

rootpath = config['rootpath']
fs = config['fs']
downsample_rate = config['downsample_rate']
active_chan_path = config['active_chan_path']
seg_to_samples = config['seg_to_samples']
dataset_out_dir = config['dataset_out_dir']
if os.path.exists(dataset_out_dir):
    os.system("rm -rf {}".format(dataset_out_dir))
os.makedirs(dataset_out_dir, exist_ok=True)

def transform2UnixTime(time_str: str):
    if time_str.count(':') == 2:
        hour, min, sec = time_str.strip().split(':')
    else:
        assert len(time_str.strip()) == 6, "'time' format must be hour(2)-minute(2)-second(2)"
        hour, min, sec = time_str[:2], time_str[2:4], time_str[4:6]
    date_time = datetime.datetime(2023, 1, 1, int(hour), int(min), int(sec), 0, pytz.timezone('Etc/GMT-8'))

    return datetime.datetime.timestamp(date_time)

def read_edf_file(edf_path):
    raw_data = mne.io.read_raw_edf(edf_path, preload=False)
    eeg = raw_data.to_data_frame() # pd.DataFrames
    return eeg

def load_active_channels(txt_file):
    active_channels = dict()
    with open(txt_file, 'r') as f:
        for line in f.readlines():
            if line.strip():
                num, chan = line.strip().split(': ')
                active_channels[num] = chan
    return active_channels

def extract_timestamps(subjects: list):
    timestamps_dict = dict()
    for subject_dir in subjects:
        edfs = []
        timestamps_file = None
        for p in os.listdir(subject_dir):
            if p.endswith('.edf'):
                edfs.append(p)
            elif p.endswith('.txt') and '时间戳记录' in p:
                timestamps_file = os.path.join(subject_dir, p)
        assert timestamps_file is not None
        # edfs.sort(key = lambda x: x.strip().split('.'))
        with open(timestamps_file, 'r') as f:
            for i, line in enumerate(f.readlines()):
                if i != 0 and line.strip():
                    num, eeg_start, vid_start, exp_start, exp_end, exp_type = line.strip().split()[:6]
                    
                    task_name = '{}实验{}.edf'.format(exp_type, num)
                    if task_name not in edfs:
                        task_name = '实验{}-{}.edf'.format(num, exp_type)    
                    assert task_name in edfs
                    timestamps_dict[os.path.join(subject_dir, task_name)] = dict(
                        eeg_start=eeg_start, vid_start=vid_start, exp_start=exp_start,
                        exp_end=exp_end
                    )
                    LOGGER.info('{} >> eeg_start: {}, vid_start: {}, exp_start: {}, exp_end: {}'.format(task_name, eeg_start,
                                vid_start, exp_start, exp_end))
    return timestamps_dict

def extract_seg(edf_pth, seg_info: dict, active_channels: dict):
    cur_eeg_all = read_edf_file(edf_pth) # dataframe
    columns = list(cur_eeg_all.columns)
    for col in columns:
        if col not in active_channels:
            cur_eeg_all.drop(columns=col, inplace=True)
    for i in range(len(cur_eeg_all.columns)):
        assert cur_eeg_all.columns[i] in active_channels
        cur_eeg_all.rename(columns={cur_eeg_all.columns[i]: active_channels[cur_eeg_all.columns[i]]}, inplace=True)
        # cur_eeg_all.columns[i] = active_channels[cur_eeg_all.columns[i]] # 换名

    # breakpoint()
    eeg_start = transform2UnixTime(seg_info['eeg_start'])
    vid_start = transform2UnixTime(seg_info['vid_start'])
    exp_start = transform2UnixTime(seg_info['exp_start'])
    exp_end = transform2UnixTime(seg_info['exp_end'])
    LOGGER.info(f'unix_time >> eeg_start: {eeg_start}, vid_start: {vid_start}, exp_start: {exp_start}, exp_end: {exp_end}')

    valid_eeg = cur_eeg_all.to_numpy()[int(exp_start-eeg_start)*fs:int(exp_end-eeg_start)*fs]
    LOGGER.info("segment length: {} sec".format(valid_eeg.shape[0] / fs))
    return valid_eeg

def create_dataset():
    for k, v in config.items():
        LOGGER.info('{}: {}'.format(k, v))

    breakpoint()
    subjects = [os.path.join(rootpath, p) for p in os.listdir(rootpath) if os.path.isdir(os.path.join(rootpath, p))]
    active_channels = load_active_channels(active_chan_path)
    LOGGER.info('active_channels: {}'.format(str(active_channels)))

    window_length = int(seg_to_samples['window_length'] * fs)
    stepsize = int(seg_to_samples['stepsize'] * fs)
    LOGGER.info('window length: {}, stepsize: {}'.format(window_length, stepsize))
    label_dict = {'低负荷': 0, '中负荷': 1, '高负荷': 2}
    total_dict = {'低负荷': 0, '中负荷': 0, '高负荷': 0}
    sub_timestamp_infos = extract_timestamps(subjects)
    total = 0
    for i_seg, (edf_path, info) in enumerate(sub_timestamp_infos.items()):
        subject = edf_path.split('/')[-2].split('_')[0]
        edf_name = edf_path.split('/')[-1]
        label = None
        for k in label_dict:
            if k in edf_name:
                label = label_dict[k]
                break
        assert label is not None

        LOGGER.info('extracting exp {}...'.format(edf_name))
        # breakpoint()
        cur_seg = extract_seg(edf_path, info, active_channels) # np.array
        assert cur_seg.shape[0] > window_length
        end_point = window_length
        while end_point <= cur_seg.shape[0]:
            sample = cur_seg[end_point-window_length:end_point:downsample_rate, :]
            end_point += stepsize
            sample_name = str(total).zfill(5) + '_{}_{}_{}.txt'.format(subject, i_seg, label)
            np.savetxt(os.path.join(dataset_out_dir, sample_name), sample)
            total += 1
            if label == 0:
                total_dict['低负荷'] += 1
            elif label == 1:
                total_dict['中负荷'] += 1
            elif label == 2:
                total_dict['高负荷'] += 1
    LOGGER.info('Produce a total of {} samples, {}'.format(total, str(total_dict)))

if __name__ == '__main__':
    create_dataset()