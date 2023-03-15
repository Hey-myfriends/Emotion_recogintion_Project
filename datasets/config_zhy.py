config = dict(
    rootpath = '/home/bebin.huang/Code/emotion_recognition/脑电数据/',
    active_chan_path = '/home/bebin.huang/Code/emotion_recognition/脑电数据/通道对应关系.txt',
    fs = 1000,
    downsample_rate = 2,

    seg_to_samples = dict(
        window_length = 4, # sec
        stepsize = 0.5
    ),
    dataset_out_dir = '/home/bebin.huang/Code/emotion_recognition/Datasets',
    train_seg = [0, 1, 2, 6, 7, 8, 9, 10, 11],
    test_seg = [3, 4, 5, 12, 13, 14],

    suffix = 'zhy')
