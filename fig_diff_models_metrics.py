import numpy as np
import matplotlib.pyplot as plt

DGCNN_LSTM = [73.65,	68.23,	75.42,	70.29,	76.15]
_4DRCNN = [70.38,	65.44,	72.26,	62.90,	70.85]
xticks = ['Accu', 'Sens', 'Spec', 'Prec', 'AUC']

x, width = np.arange(len(xticks)), 0.25
fig, ax = plt.subplots(1, 1, figsize=(12, 9))
ax.bar(x-width/2, DGCNN_LSTM, width, label='DGCNN_LSTM', color='orange')
ax.bar(x+width/2, _4DRCNN, width, label='4DRCNN', color='g', hatch='//')
ax.set_ylabel(ylabel='Percent', fontsize=15)
# ax.set(ylabel='Percent')
ax.legend(fontsize=15)
plt.xticks(ticks=list(range(len(xticks))), labels=xticks, fontsize=17)
plt.yticks(ticks=[0, 20, 40, 60, 80, 100], fontsize=12)
ax.set_ylim([0, 105])
ax.grid(alpha=0.5, axis='y')

fig.savefig('./diff_models_metrics.png', dpi=300, bbox_inches='tight')
plt.close()