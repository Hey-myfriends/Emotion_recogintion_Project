

import torch, random, os, time, json, glob
import datetime
import numpy as np
from models import build_model2
from datasets import build_dataset, collate_fn
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
        self.batchsize = 128
        self.epochs = 50
        self.clip_max_norm = 0.3
        self.lr_drop = 15
        self.gamma = 0.5
        self.num_cls = 2
        self.tag = "total_epochs_{}_DGCNN_LSTM2_{}".format(self.epochs, self.num_cls)
        self.level = "episode" #"sample"
        self.output_dir = "./outputs_{}_{}".format(self.level, self.tag)
        if log_:
            self.log()

    def log(self):
        # if os.path.exists(self.output_dir):
        #     os.system('rm -rf {}'.format(self.output_dir))
        logger.info(f"epoches: {self.epochs}, bs: {self.batchsize}, num_cls: {self.num_cls}, remove the relu of final classifier")

def main():
    # pdb.set_trace()
    args = Arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'ckpts'), exist_ok=True) 
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    in_chan, K, out_chan, d, hidden_size = 16, 8, 32, 5, 128
    cls_weights = torch.Tensor([1.0, 5.0])
    logger.info(f"in_chan: {in_chan}, out_chan: {out_chan}, d: {d}, num_class: {args.num_cls}, lstm_hid_size: {hidden_size}")

    dataset_train, dataset_val = build_dataset(feat_to_4D=False, multiclass= True if args.num_cls >= 3 else False)

    data_loader_train = DataLoader(dataset_train, args.batchsize, shuffle=True, collate_fn=collate_fn)
    data_loader_val = DataLoader(dataset_val, args.batchsize, shuffle=False, collate_fn=collate_fn)
    data_loader_val_shuffle = DataLoader(dataset_val, args.batchsize, shuffle=True, collate_fn=collate_fn)

    # model, criterion = build_model(in_channels, out_channels, kernels, fc_num, hidden_size, args.num_cls, dataset_train.cls_weights)
    model, criterion = build_model2(in_chan, K, out_chan, d, args.num_cls, hidden_size, dataset_train.cls_weights)
    logger.info("Model: \n{}".format(str(model)))
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
    # breakpoint()
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, args.device, 
            epoch, args.num_cls, args.clip_max_norm
        )
        lr_schedular.step()

        test_stats = evaluate(
            model, criterion, data_loader_val_shuffle, args.device, args.num_cls
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

    fields = ["loss", "loss_ce"]
    for i in range(args.num_cls):
        fields.append('loss_{}'.format(i))
    fields.append('class_error')
    plot_logs(args.output_dir, log_name=logpath.split('/')[-1], fields=tuple(fields))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time cost: {}'.format(total_time_str))

    if len(ckpts) == 0:
        ckpts = glob.glob(os.path.join(args.output_dir, 'ckpts/*.pth'))
    performance = test(model, data_loader_val, args.output_dir, args.device, ckpts=ckpts, level=args.level, multiclass= args.num_cls >= 3)
    with open(os.path.join(args.output_dir, "test.json"), "w") as pf:
        pf.write(json.dumps(performance, ensure_ascii=False, cls=JsonEncoder, indent=4, separators=(",", ":")))

    if os.path.exists('train_test.log'):
        os.system('mv {} {}'.format('train_test.log', args.output_dir))

@torch.no_grad()
def test(model: torch.nn.Module, val_dataloader: Iterable, output_dir: str, device: torch.device, ckpts: list, level='episode', multiclass=False):
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
        predictions = torch.cat(predictions, dim=0).softmax(axis=1).numpy()
        targets = torch.cat(targets, dim=0).numpy()
        preds = predictions.argmax(axis=1)

        # breakpoint()
        accu = metrics.accuracy_score(targets, preds)
        f1_score = metrics.f1_score(targets, preds, average = 'binary' if not multiclass else 'weighted')
        sens = metrics.recall_score(targets, preds, average = 'binary' if not multiclass else 'weighted')
        prec = metrics.precision_score(targets, preds, average = 'binary' if not multiclass else 'weighted')
        auc = metrics.roc_auc_score(targets, predictions, multi_class= 'raise' if not multiclass else 'ovr')
        # auc = None
        cmat = metrics.confusion_matrix(targets, preds) # dim 0: gt; dim 1: dt, label 0, 1, ... along axis
        # spec = cal_spec(cmat)
        # geom = np.sqrt(sens * spec)
        samples_level = {"accu": accu, "sens": sens, "prec": prec,
                        "auc": auc, "f1_score": f1_score, "cmat": cmat.tolist()}
        logger.info("Epoch {} sample level performance: ".format(ckpt["epoch"]) + str(samples_level))
        performance.update({"Epoch_{}_sampleLevel".format(ckpt["epoch"]): samples_level})
    return performance

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