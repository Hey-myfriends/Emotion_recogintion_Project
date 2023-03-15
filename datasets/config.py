
config = dict(
    rootpath = "/home/bebin.huang/Code/emotion_recognition/高负荷生理采集实验-1023/04 EGG数据/",
    fs = dict(eeg = 500, physio = 100),
    eeg_start_time = [131629, 131918, 132158, 140808, 141131, 141511, 141904, 151638, 151940, 154517, 161010, 161349, 172246, 144413, 144930, 154517, 155204, 161010, 161349, 172246],
    eeg_start_shift = [44, 26, 43, 42, 40, 31,  31, 35, 34, 30, 29, 27, 29, 35, [30, 192], 147, [0, 123], 135, 190, 202],
    length =          [75, 85, 74, 89, 96, 103, 83, 86, 95, 61, 61, 66, 56, 80, [73, 57],  30,  [62, 12], 40,  37,  15],
    seg_labels =      [0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  1,  1,         1,   1,        1,   1,   1],

    seg_to_samples = dict(
        window_length = 4, # sec
        stepsize = 1
    ),
    dataset_out_dir = '/home/bebin.huang/Code/emotion_recognition/Datasets',

    train_seg = [0, 1, 3, 4, 5, 7, 9, 10, 11, 12, 13, '14_1', 15, '16_0', 17, 18],
    test_seg = [2, 6, 8, '14_0', '16_1', 19]
)