import numpy as np
from scipy.io import loadmat

file_path='/Users/anayisi/Documents/Master/Emotion recognition/SEED/SEED/Preprocessed_EEG/'
people_name = ['1_20131027', '1_20131030', '1_20131107',
               '6_20130712', '6_20131016', '6_20131113',
               '7_20131027', '7_20131030', '7_20131106',
               '15_20130709', '15_20131016', '15_20131105',
               '12_20131127', '12_20131201', '12_20131207',
               '10_20131130', '10_20131204', '10_20131211',
               '2_20140404', '2_20140413', '2_20140419',
               '5_20140411', '5_20140418', '5_20140506',
               '8_20140511', '8_20140514', '8_20140521',
               '13_20140527', '13_20140603', '13_20140610',
               '3_20140603', '3_20140611', '3_20140629',
               '14_20140601', '14_20140615', '14_20140627',
               '11_20140618', '11_20140625', '11_20140630',
               '9_20140620', '9_20140627', '9_20140704',
               '4_20140621', '4_20140702', '4_20140705']
short_name = ['djc', 'djc', 'djc', 'mhw', 'mhw', 'mhw', 'phl', 'phl', 'phl',
              'zjy', 'zjy', 'zjy', 'wyw', 'wyw', 'wyw', 'ww', 'ww', 'ww',
              'jl', 'jl', 'jl', 'ly', 'ly', 'ly', 'sxy', 'sxy', 'sxy',
              'xyl', 'xyl', 'xyl', 'jj', 'jj', 'jj', 'ys', 'ys', 'ys',
              'wsf', 'wsf', 'wsf', 'wk', 'wk', 'wk', 'lqj', 'lqj', 'lqj']
final_data = np.empty([0, 62, 1000])
for i in range(len(people_name)):
    file_name = file_path + people_name[i]
    data = loadmat(file_name)
    before_final = np.empty([15, 62, 1000])
    for trial in range(15):
        tmp_trial_signal = data[short_name[i] + '_eeg' + str(trial + 1)][:, 0:37000:37]
        before_final[trial, :, :] = tmp_trial_signal
    final_data = np.vstack([final_data, before_final])
print(final_data.shape)
