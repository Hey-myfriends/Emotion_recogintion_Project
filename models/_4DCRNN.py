import sys
def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
    print(sys.path[0])
add_path('/home/bebin.huang/Code/emotion_recognition/code')
import torch
from torch import nn
import torch.nn.functional as F
import sklearn.metrics as metrics
from get_logger import get_root_logger, logger

class _4DCRNN(nn.Module):
    def __init__(self, in_channels=4, out_channels: list =[64], kernels:list = [5], 
                fc_num=512, hidden_size=128, num_cls=2) -> None:
        super().__init__()
        CNN_learner = []
        assert isinstance(out_channels, list) and isinstance(kernels, list)
        assert len(out_channels) == len(kernels)
        for oup, ker in zip(out_channels, kernels):
            assert ker % 2 == 1
            CNN_learner.append(nn.Conv2d(in_channels, oup, kernel_size=ker, padding=ker//2))
            # CNN_learner.append(nn.BatchNorm2d(oup))
            CNN_learner.append(nn.ReLU())
            in_channels = oup
        CNN_learner.append(nn.MaxPool2d(kernel_size=2, stride=2))
        CNN_learner.append(nn.Flatten(start_dim=1))
        # CNN_learner.append(nn.Linear(1280, 512))
        self.CNN_learner = nn.Sequential(*CNN_learner)

        self.flatten_dim = 1024
        self.fc = nn.Linear(self.flatten_dim, fc_num)
        self.lstm = nn.LSTM(input_size=fc_num, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.classifer = nn.Linear(hidden_size, num_cls)

    def forward(self, input):
        '''
        Parameters:
        input: shape torch.size([bs, L, d, h, w])
        where h*w is the number of channels expanded in spatial space, 
        d is the number of subbands and L is the number of sub-windows in a EEG segment.
        '''
        # breakpoint()
        bs, L, d, h, w = input.shape
        input = input.view(bs*L, d, h, w)
        features = self.CNN_learner(input) # [bs*L, out_chan * h * w]
        if self.flatten_dim != features.shape[-1]:
            logger.warning("Warning! "+f"FC in_features error, {features.shape[-1]} != {self.fc.in_features}, Replace the fc layer with new input dim.")
            self.flatten_dim = features.shape[-1]
            out_fc_dim = self.fc.out_features
            self.fc = nn.Linear(self.flatten_dim, out_fc_dim).to(input.device)
            logger.info("Model: \n{}".format(str(self)))
        # assert features.shape[-1] == self.fc.in_features, f"FC in_features error, {features.shape[-1]} != {self.fc.in_features}"
        features = self.fc(features)
        features = features.view(bs, L, -1) # [bs, L, H_in]
        out_feat, (h_n, c_n) = self.lstm(features) # out_feat in shape [bs, L, H_in], h_n in shape [1, bs, H_in] and out_feat[:, -1] == h_n[0]
        logits = self.classifer(out_feat[:, -1])
        return logits

class Criterion(nn.Module):
    def __init__(self, cls_weight, weight_dict={'loss_ce': 1}) -> None:
        super().__init__()
        self.cls_weights = cls_weight
        self.weight_dict = weight_dict
        logger.info("cls_weight: {}, weight_dict: {}".format(str(self.cls_weights), str(self.weight_dict)))

    def forward(self, outputs, targets):
        loss_ce = F.cross_entropy(outputs, targets, reduction="none")
        self.cls_weights = self.cls_weights.to(outputs.device)
        at = self.cls_weights.gather(dim=0, index=targets)
        loss_ce = at * loss_ce
        losses = {"loss_ce": loss_ce.mean(), 
                    "loss_0": loss_ce[targets == 0].mean(),
                    "loss_1": loss_ce[targets == 1].mean(), 
                    'class_error': 100 - accuracy(outputs, targets) * 100}
        return losses

@torch.no_grad()
def accuracy(outputs, targets):
    _, pred = outputs.max(dim=1)
    accu = metrics.accuracy_score(targets.cpu().numpy(), pred.cpu().numpy())
    return accu

def build_model(in_channels, out_channels, kernels, fc_num, hidden_size, num_cls, cls_weights):
    model = _4DCRNN(in_channels=in_channels, out_channels=out_channels, kernels=kernels,
            fc_num=fc_num, hidden_size=hidden_size, num_cls=num_cls)
    criterion = Criterion(cls_weights)
    return model, criterion

if __name__ == "__main__":
    out_channels = [64, 128, 256, 64]
    kernels = [5, 3, 3, 1]
    fc_num, hidden_size, num_cls = 512, 128, 2
    model = _4DCRNN(in_channels=4, out_channels=out_channels, kernels=kernels,
            fc_num=fc_num, hidden_size=hidden_size, num_cls=num_cls)
    breakpoint()
    input = torch.rand(32, 50, 4, 16, 16)
    logits = model(input)
    breakpoint()
    print(logits.shape)