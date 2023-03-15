import torch
from torch import nn
import torch.nn.functional as F
from ._4DCRNN import Criterion

'''
网络执行顺序：
初始化邻接矩阵 --- 求出拉普拉斯矩阵 --- 生成切比雪夫多项式项 --- 执行GraphConvolution
--- 融合特征 --- 分类
'''

class DGCNN_LSTM(nn.Module):
    def __init__(self, in_chan, K, out_chan, d, num_cls=2, hidden_size=128) -> None: # d is the number of subbands
        super().__init__()
        self.in_chan = in_chan
        self.W = nn.Parameter(torch.rand(in_chan, in_chan))
        nn.init.xavier_normal_(self.W)

        self.chebynet = ChebyNet(K, in_chan, out_chan)
        self.flatten = nn.Flatten(start_dim=2)
        self.lstm = nn.LSTM(input_size=d*out_chan, hidden_size=hidden_size, num_layers=1, batch_first=True)
        # self.classifier = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(), nn.Linear(hidden_size, num_cls))
        self.classifier = nn.Linear(hidden_size, num_cls)

    def forward(self, x):
        '''
        x: torch.Tensor([bs, L, d, num_chans]), L is the number of sub-windows within a window, d is the number of subbands
        '''
        L = normalize_A(self.W)
        out = self.chebynet(x, L) # [bs, L, d, out_chan]
        out = self.flatten(out)
        out_feat, (c_n, h_n) = self.lstm(out)
        logits = self.classifier(out_feat[:, -1])
        return logits

class ChebyNet(nn.Module):
    def __init__(self, K: int, in_chan, out_chan) -> None:
        super().__init__()
        assert K > 0
        self.K = K # the order of Chebyshev polynomials
        self.gcnn = nn.ModuleList([GraphConvolution(in_chan, out_chan) for _ in range(K)])
        self.relu = nn.LeakyReLU()
        

    def forward(self, x, L):
        '''
        x: input
        L: Laplacian matrix
        '''
        polyn_items = generate_cheby_polynomial(L, self.K)
        results = None
        for i, layer in enumerate(self.gcnn):
            if i == 0:
                results = layer(x, polyn_items[i])
            else:
                results += layer(x, polyn_items[i])
        return self.relu(results)

class GraphConvolution(nn.Module):
    def __init__(self, in_chan, out_chan) -> None:
        super().__init__()
        self.linear = nn.Linear(in_chan, out_chan, bias=False)

    def forward(self, x, polyn_item):
        out = F.linear(x, polyn_item, bias=None)
        out = self.linear(out)
        return out

def generate_cheby_polynomial(A, K):
    support = []
    for i in range(K):
        if i == 0:
            support.append(torch.eye(A.shape[1]).to(A.device))
        elif i == 1:
            support.append(A)
        else:
            temp = torch.matmul(support[-1], A)
            support.append(temp)
    return support

def normalize_A(A, symmetry=False):
    A = F.relu(A) # adj matrix must be non-negative
    if symmetry:
        A = A + torch.transpose(A, 0, 1)
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    else:
        d = torch.sum(A, 1)
        d = 1 / torch.sqrt(d + 1e-10)
        D = torch.diag_embed(d)
        L = torch.matmul(torch.matmul(D, A), D)
    return L

def build_model2(in_chan, K, out_chan, d, num_cls, hidden_size, cls_weights):
    model = DGCNN_LSTM(in_chan, K, out_chan, d, num_cls=num_cls, hidden_size=hidden_size)
    criterion = Criterion(cls_weights)
    return model, criterion

if __name__ == "__main__":
    in_chan, K, out_chan, d, num_cls, hidden_size = 32, 8, 32, 5, 2, 128
    model = DGCNN_LSTM(in_chan, K, out_chan, d, num_cls=num_cls, hidden_size=hidden_size)

    x = torch.rand(64, 8, d, in_chan)
    y = model(x)
    print(y.shape)