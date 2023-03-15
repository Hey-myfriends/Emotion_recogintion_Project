

import torch, random, os, time, json
import datetime
import numpy as np
from models import build_model
from datasets import build_dataset, collate_fn, split_n_fold
from utils.plot_utils import plot_logs
from torch.utils.data import DataLoader
from engine import train_one_epoch, evaluate
from get_logger import get_root_logger, logger
from jsonEncoder import JsonEncoder
from typing import Iterable
from tqdm import tqdm
import sklearn.metrics as metrics
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# logger = get_root_logger(level=logging.DEBUG, console_out=True, logName="log.log")

class Arguments(object):
    def __init__(self, log_=True) -> None:
        logger.info(f"This machine has {torch.cuda.device_count()} gpu...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = 10086
        self.batchsize = 32
        self.epochs = 50
        self.clip_max_norm = 0.3
        self.lr_drop = 15
        self.gamma = 0.5
        self.tag = "total_epochs_{}_normalized_cls_weight(1, 5)".format(self.epochs)
        self.level = "episode" #"sample"
        self.output_dir = "./outputs_{}_{}".format(self.level, self.tag)
        if log_:
            self.log()

    def log(self):
        logger.info(f"epoches: {self.epochs}, bs: {self.batchsize}")

def main():
    # pdb.set_trace()
    args = Arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ckpts'), exist_ok=True) 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    in_channels, out_channels = 5, [64, 128, 256, 64]
    kernels = [5, 3, 3, 1]
    fc_num, hidden_size, num_cls = 256, 128, 2
    cls_weights = torch.Tensor([1.0, 5.0])
    logger.info(f"in_chan: {in_channels}, out_chan: {out_channels}, kernels: {kernels}, num_class: {num_cls}, fc_num: {fc_num}, lstm_hid_size: {hidden_size}")

    dataset_train, dataset_val = build_dataset()

    data_loader_train = DataLoader(dataset_train, args.batchsize, shuffle=True, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, args.batchsize, shuffle=False, collate_fn=collate_fn)
    data_loader_val_shuffle = DataLoader(dataset_val, args.batchsize, shuffle=True, collate_fn=collate_fn)

    # model, criterion = build_model(in_channels, out_channels, kernels, fc_num, hidden_size, num_cls, dataset_train.cls_weights)
    model, criterion = build_model(in_channels, out_channels, kernels, fc_num, hidden_size, num_cls, cls_weights)
    model.to(args.device)
    criterion.to(args.device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]}, 
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], 
        "lr": 1e-3}
        ]
    optimizer = torch.optim.Adam(param_dicts, lr=1e-3, weight_decay=1e-4)
    lr_schedular = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=args.gamma)
    
    logger.info("Start training...")
    start_time = time.time()
    ckpts = []
    logpath = os.path.join(args.output_dir, 'log.txt')
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, args.device, 
            epoch, args.clip_max_norm
        )
        lr_schedular.step()

        test_stats = evaluate(
            model, criterion, data_loader_val_shuffle, args.device
        )

        log_stats = {"epoch": epoch,
                    "n_params": n_parameters,
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"test_{k}": v for k, v in test_stats.items()},
                    }

        if args.output_dir:
            if epoch >= 0:
                ckpt = os.path.join(args.output_dir, f"ckpts/checkpoint_{epoch:04}.pth")
                torch.save({
                    # "fold": fold,
                    "epoch": epoch,
                    "args": args,
                    "model": model.state_dict(),
                    # "stats": log_stats,
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_schedular.state_dict()
                }, ckpt)
                ckpts.append(ckpt)
            with open(logpath, "a") as f:
                f.write(json.dumps(log_stats) + "\n")

    plot_logs(args.output_dir, log_name=logpath.split('/')[-1], fields=("loss", "loss_ce", "loss_0", "loss_1", "class_error"))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time cost: {}'.format(total_time_str))

    performance = test(model, data_loader_val, args.output_dir, args.device, ckpts=ckpts, level=args.level)
    with open(os.path.join(args.output_dir, "test.json"), "w") as pf:
        pf.write(json.dumps(performance, ensure_ascii=False, cls=JsonEncoder, indent=4, separators=(",", ":")))

@torch.no_grad()
def test(model: torch.nn.Module, val_dataloader: Iterable, output_dir: str, device: torch.device, ckpts: list, level='episode'):
    performance = {}
    model = model.to(device)
    model.eval()

    for c in ckpts:
        ckpt = torch.load(c)
        model.load_state_dict(ckpt["model"])
        pbar = tqdm(val_dataloader)

        predictions, targets = [], []
        for samples, t in pbar:
            samples = samples.to(device)
            targets.append(t)

            outputs = model(samples)
            predictions.append(outputs.cpu())
        predictions = torch.cat(predictions, dim=0).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        preds = predictions.argmax(axis=1)

        accu = metrics.accuracy_score(targets, preds)
        f1_score = metrics.f1_score(targets, preds)
        sens = metrics.recall_score(targets, preds)
        prec = metrics.precision_score(targets, preds)
        auc = metrics.roc_auc_score(targets, preds)
        # auc = None
        cmat = metrics.confusion_matrix(targets, preds) # dim 0: gt; dim 1: dt, label 0, 1, ... along axis
        spec = cal_spec(cmat)
        geom = np.sqrt(sens * spec)
        samples_level = {"accu": accu, "sens": sens, "spec": spec, "prec": prec, "geom": geom, 
                        "auc": auc, "f1_score": f1_score, "cmat": cmat.tolist()}
        logger.info("Epoch {} sample level performance: ".format(ckpt["epoch"]) + str(samples_level))
        performance.update({"Epoch_{}_sampleLevel".format(ckpt["epoch"]): samples_level})
    return performance

def cal_spec(cmat: np.array):
    return cmat[0, 0] / (cmat[0, 0] + cmat[0, 1])

if __name__ == "__main__":
    main()
    # args = Arguments()
    # plot_logs(args.output_dir, log_name="train.txt", fields=("loss", "loss_ce", "loss_NW", "loss_preFoG", "loss_FoG", "class_error"))
    # test_ckpts()

    # import glob
    # args = Arguments(log_=False)
    # # sam_sel = glob.glob(os.path.join(args.output_dir, "011_003_000_*"))
    # sam_sel = [p for p in os.listdir(args.output_dir) if p.startswith("011_003_")]
    # print(sam_sel)