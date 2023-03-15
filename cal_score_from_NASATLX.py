import os
import pandas as pd
import numpy as np

xlsx_files = ['../脑电数据/唐_block1/NASA-TLX任务工作负荷评估问卷_20230227.xlsx', '../脑电数据/陈_block1/NASA-TLX任务工作负荷评估问卷-20230302.xlsx']
keys = ['脑力需求', '体力需求', '时间需求', '绩效水平', '努力程度', '受挫程度']

# breakpoint()
scores = dict(low=[], mid=[], high=[])
for i in range(len(xlsx_files)):
    table_cur = pd.read_excel(xlsx_files[i])
    if i == 0:
        table_cur.drop(i, inplace=True)
    print(table_cur.columns)

    weights = {k: 0 for k in keys}
    scores_now = []
    # breakpoint()
    for row in range(table_cur.shape[0]):
        scr = np.array([table_cur.iat[row, s] for s in range(7, 7+len(keys))])
        for col in range(13, 28):
            term = table_cur.iat[row, col]
            assert term in keys
            weights[term] += 1
        # breakpoint()
        num = sum(weights[k] for k in weights)
        weight_now = np.array([weights[k] for k in weights]) / num
        score = (scr * weight_now * 5).sum()
        scores_now.append(score)
        if '低' in table_cur.iat[row, -2]:
            scores['low'].append(score)
        elif '中' in table_cur.iat[row, -2]:
            scores['mid'].append(score)
        elif '高' in table_cur.iat[row, -2]:
            scores['high'].append(score)
# breakpoint()
for k in scores:
    print("scores {}: {}\n".format(k, sum(scores[k])/len(scores[k])))
