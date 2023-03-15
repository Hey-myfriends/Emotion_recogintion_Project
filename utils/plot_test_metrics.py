import numpy as np
import matplotlib.pyplot as plt
import json, os

def plot_test_metrics(metric_path, fields=('accu', 'sens', 'spec', 'geom', 'auc')):
    with open(metric_path, 'r') as f:
        metrics_each_epoch = json.load(f)

    metrics_all = dict(epoch = [])
    for k, v in metrics_each_epoch.items():
        metrics_all['epoch'].append(eval(k.split('_')[1]))
        for kk, vv in v.items():
            if kk not in metrics_all:
                metrics_all[kk] = []
            metrics_all[kk].append(vv)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    epoch = np.array(metrics_all['epoch'])
    for field in fields:
        if field not in metrics_all:
            continue
        ax.plot(epoch, np.array(metrics_all[field])*100, label=field, linewidth=2, marker='.')
    ax.set(xlabel='Epoch', ylabel='%')
    ax.legend()
    ax.grid(axis='y')
    # parent_path = os.path.abspath(os.path.join(metric_path, os.pardir))
    fig.savefig(metric_path.replace('json', 'png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    metrics_path = './outputs_episode_total_epochs_50_DGCNN_LSTM2/test.json'
    plot_test_metrics(metrics_path)