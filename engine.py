from typing import Iterable
import torch, math, sys, pdb
from tqdm import tqdm
from collections import deque, defaultdict
from get_logger import logger

class SmootheValue(object):
    def __init__(self, window_size=20) -> None:
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count if self.count != 0 else 0

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
    data_loader: Iterable, optimizer: torch.optim.Optimizer,
    device: torch.device, epoch: int, num_cls: int, max_norm: float = 0):
    
    model.train()
    criterion.train()
    header = 'Epoch: [{}] Training: '.format(epoch)
    meters = defaultdict(SmootheValue)
    meters["loss"] = SmootheValue(window_size=20)
    meters["lr"] = SmootheValue(window_size=1)
    pbar = tqdm(data_loader)

    for samples, targets in pbar:
        samples = samples.to(device)
        targets = targets.to(device)

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        # pdb.set_trace()
        for k in loss_dict.keys():
            if k not in meters.keys():
                meters[k] = SmootheValue(window_size=20)
            meters[k].update(loss_dict[k].cpu().item() if isinstance(loss_dict[k], torch.Tensor) else loss_dict[k])
            
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k]*weight_dict[k] for k in loss_dict.keys() if k in weight_dict.keys())

        loss_value = losses.cpu().item()

        if not math.isfinite(loss_value):
            logger.error(f"loss is {loss_value}, stop training...")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        # breakpoint()
        all_grad = {name: params.grad.data.abs() for name, params in model.named_parameters() if params.requires_grad}
        for k, v in all_grad.items():
            if v.max().item() > max_norm:
                logger.info(f"{k}: {[v.mean().item(), v.max().item()]}")
        # print({k: [v.mean().item(), v.max().item()] for k, v in all_grad.items()})
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        meters["loss"].update(loss_value)
        meters["lr"].update(optimizer.param_groups[0]["lr"])
        msg = "Train-> Epoch: {}, lr:  {:.5f} loss: {:.4f} loss_ce: {:.4f}, class_error: {:.2f}%".format(
            epoch, meters["lr"].value, meters["loss"].avg, meters["loss_ce"].avg, meters["class_error"].avg
        )
        for i in range(num_cls):
            msg += ", loss_{}: {:.4f}".format(i, meters['loss_{}'.format(i)].avg)
        pbar.set_description(msg)

    stats = {k: meter.global_avg for k, meter in meters.items()}
    logger.info(header + str(stats))
    return stats

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module,
            data_loader: Iterable, device: torch.device, num_cls: int):
    # pdb.set_trace()
    model.eval()
    criterion.eval()
    meters = defaultdict(SmootheValue)
    meters["loss"] = SmootheValue(window_size=20)
    pbar = tqdm(data_loader)

    for samples, targets in pbar:
        samples = samples.to(device)
        targets = targets.to(device)

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        for k in loss_dict.keys():
            if k not in meters.keys():
                meters[k] = SmootheValue(window_size=20)
            meters[k].update(loss_dict[k].cpu().item() if isinstance(loss_dict[k], torch.Tensor) else loss_dict[k])

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k]*weight_dict[k] for k in loss_dict.keys() if k in weight_dict.keys())
        loss_value = losses.cpu().item()
        meters["loss"].update(loss_value)
        msg = "Test-> loss: {:.4f} loss_ce: {:.4f}, class_error: {:.2f}%".format(
            meters["loss"].avg, meters["loss_ce"].avg, meters["class_error"].avg
        )
        for i in range(num_cls):
            msg += ", loss_{}: {:.4f}".format(i, meters['loss_{}'.format(i)].avg)
        pbar.set_description(msg)
    stats = {k: meter.global_avg for k, meter in meters.items()}
    logger.info("Testing: " + str(stats) + "\n")
    return stats
