# sample saving format: num_seg_subseg_label.txt
import sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
    print(sys.path[0])
add_path('/home/bebin.huang/Code/emotion_recognition/Emotion-Recognition-with-4DRCNN-and-DGCNN_LSTM')
import torch
import os, glob, random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets.config_zhy import config
from get_logger import logger
from utils import feature_extractor, feature_to_4D
from tqdm import tqdm

class ER_dataset(Dataset):
    def __init__(self, samples: list, level="episode", mode="train", feat_to_4D=True, multiclass=False) -> None:
        super().__init__()
        assert level in ['episode', 'sample']
        assert mode in ["train", "val"]
        self.dataset_out_dir = config['dataset_out_dir']
        self.samples = samples
        self.level = level
        self.mode = mode
        self.multiclass = multiclass
        self.feat_to_4D = feat_to_4D
        # breakpoint()
        if self.level == 'episode':
            sam_all = []
            for sam in samples:
                sam_all.extend([p for p in glob.glob(os.path.join(self.dataset_out_dir, '*_*_{}_*.txt'.format(sam)))])
            self.samples = sam_all
            # breakpoint()
            if not self.multiclass:
                self.samples = [s for s in self.samples if s.split('.')[-2].split('_')[-1] != '1']
        self.cls_weights = self.cal_cls_weight()

    def cal_cls_weight(self):
        total = len(self.samples)
        t0, t1, t2 = 0, 0, 0
        for sam in self.samples:
            lab = sam.split('.')[-2].split('_')[-1]
            if lab == '0': t0 += 1
            elif lab == '1': t1 += 1
            elif lab == '2': t2 += 1
            else: raise ValueError("Label error: must be 0, 1 or 2.")
        # breakpoint()
        assert total == t0 + t1 + t2
        if t1 == 0:
            ratio = [t2/(t0+t2), t0/(t0+t2)]
        else:
            ratio = [t1*t2, t0*t2, t0*t1]
            ratio = [n/sum(ratio) for n in ratio]
        logger.info("Mode = {}, total = {}, [t0, t1, t2] = [{}, {}, {}], ratio = {}".format(
            self.mode, total, t0, t1, t2, str(ratio)
        ))
        return torch.Tensor(ratio)

    def __getitem__(self, index):
        # breakpoint()
        data = np.loadtxt(os.path.join(self.dataset_out_dir, self.samples[index]))
        data = feature_extractor(data) # [num_chans, num_bands, 2*T]
        label = self.samples[index].split('.')[-2].split('_')[-1]
        assert label in ['0', '1', '2']
        if not self.multiclass and label == '2':
            label = '1'
        if self.feat_to_4D:
            data = feature_to_4D(data) # [h, w, d, 2*T], d = num_bands
            data = torch.FloatTensor(data).permute((3, 2, 0, 1)) # for 4DRCNN
        else:
            data = torch.FloatTensor(data).permute((2, 1, 0)) # for DGCNN_LSTM, [2*T, d, num_chans]
        return data, eval(label)


    def __len__(self):
        return len(self.samples)

def collate_fn(batch: list):
    # breakpoint()
    data, labels = [], []
    for d, l in batch:
        data.append(d)
        labels.append(l)
    data = torch.stack(data, dim=0)
    data = normalize(data)
    labels = torch.LongTensor(labels)
    return data, labels

def normalize(data: torch.Tensor):
    # for 4DRCNN
    # mu = torch.Tensor([2.6190, 2.1270, 2.1157, 2.2070, 1.8544]).view((1, 1, -1, 1, 1)).to(data.device)
    # sig = torch.Tensor([1.2011, 0.9644, 1.0048, 1.2820, 1.3156]).view((1, 1, -1, 1, 1)).to(data.device)
    # for DGCNN_LSTM
    mu = torch.Tensor([5.0694, 3.6375, 3.0530, 2.8169, 2.7789]).view((1, 1, -1, 1)).to(data.device)
    sig = torch.Tensor([2.2625, 2.1988, 2.1096, 2.5440, 2.7996]).view((1, 1, -1, 1)).to(data.device)
    # breakpoint()
    return (data - mu) / sig

def cal_normalized_factor():
    dataset_train, dataset_test = build_dataset(feat_to_4D=False)
    data_train, data_test = [], []
    for i in tqdm(range(len(dataset_train))):
        data_train.append(dataset_train.__getitem__(i)[0])
    for i in tqdm(range(len(dataset_test))):
        data_test.append(dataset_test.__getitem__(i)[0])

    breakpoint()
    data_train, data_test = torch.stack(data_train, dim=0), torch.stack(data_test, dim=0)
    mean_train, std_train = data_train.mean(dim=(0, 1, 3), keepdim=True), data_train.std(dim=(0, 1, 3), keepdim=True)
    mean_test, std_test = data_test.mean(dim=(0, 1, 3), keepdim=True), data_test.std(dim=(0, 1, 3), keepdim=True)

    data_test_normalized = (data_test - mean_train) / std_train
    print(data_test_normalized.mean(dim=(0, 1, 3)), data_test_normalized.std(dim=(0, 1, 3)))


def build_dataset(feat_to_4D=True, multiclass=True):
    assert feat_to_4D == False
    # train_samples, test_samples = train_test_split()
    # breakpoint()
    dataset_train = ER_dataset(config['train_seg'], level='episode', mode='train', feat_to_4D=feat_to_4D, multiclass=multiclass)
    dataset_test = ER_dataset(config['test_seg'], level='episode', mode='val', feat_to_4D=feat_to_4D, multiclass=multiclass)
    return dataset_train, dataset_test


if __name__ == "__main__":
    # cal_normalized_factor()
    dataset_train, dataset_test = build_dataset(feat_to_4D=False, multiclass=False)
    breakpoint()
    data, label = dataset_train.__getitem__(10)
    a = 1