import logging
import random
import sys
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dataset import PredictDataset
import mpl_scatter_density  # adds projection='scatter_density'
from matplotlib.colors import LinearSegmentedColormap
import torch


def visualize_scatterplot(args, title, predict, target, scale=100.0):
    white_viridis = LinearSegmentedColormap.from_list(
        "white_viridis",
        [
            (0, "#ffffff"),
            (1e-20, "#440053"),
            (0.2, "#404388"),
            (0.4, "#2a788e"),
            (0.6, "#21a784"),
            (0.8, "#78d151"),
            (1, "#fde624"),
        ],
        N=256,
    )

    def _scatter(x, y, subplot, threshold=None):
        ax = plt.subplot(subplot, projection="scatter_density")
        density = ax.scatter_density(x, y, cmap=white_viridis)
        plt.grid(linestyle="--")
        plt.xlabel("Target", fontsize=16)
        plt.ylabel("Prediction", fontsize=16)
        plt.xlim([0, 95])
        plt.ylim([0, 95])
        plt.plot([0, 95], [0, 95], linestyle="dotted", linewidth=3, color="black")
        if threshold:
            ax = plt.gca()
            ax.set_xlim(threshold, 95)
            ax.set_ylim(threshold, 95)
            ax.set_title("Top 20%", fontsize=20)
        else:
            ax = plt.gca()
            ax.set_title("ALL", fontsize=20)
        return density

    predict = PredictDataset.denormalize(predict)
    target = PredictDataset.denormalize(target)

    max_val = max(max(target), max(predict))
    predict /= max_val
    target /= max_val
    predict *= scale
    target *= scale

    fig = plt.figure(figsize=(12, 6))
    _scatter(target, predict, 121)
    density = _scatter(target, predict, 122, threshold=80)
    fig.colorbar(density, label="Number of points per pixel")
    plt.savefig(f"{args.save_dir}/scatter_{title}.png", bbox_inches="tight")
    plt.close()


def visualize_line_plot(args, train, test, title):
    plt.figure(figsize=(12, 6))
    plt.plot(train, label="train")
    plt.plot(test, label="val")
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xlabel("epochs")
    plt.ylabel(f"{title}")
    plt.savefig(f"{args.save_dir}/{title}.png", bbox_inches="tight")
    plt.close()


def visualize_line_plot2(args, train, test, title):
    plt.figure(figsize=(12, 6))
    plt.plot(train, label="kendalltau")
    plt.plot(test, label="spearmanr")
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.xlabel("epochs")
    plt.ylabel(f"{title}")
    plt.savefig(f"{args.save_dir}/{title}.png", bbox_inches="tight")
    plt.close()


def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def to_cuda(obj):
    if torch.is_tensor(obj):
        return obj.cuda()
    if isinstance(obj, tuple):
        return tuple(to_cuda(t) for t in obj)
    if isinstance(obj, list):
        return [to_cuda(t) for t in obj]
    if isinstance(obj, dict):
        return {k: to_cuda(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))


def get_logger():
    time_format = "%m/%d %H:%M:%S"
    fmt = "[%(asctime)s] %(levelname)s (%(name)s) %(message)s"
    formatter = logging.Formatter(fmt, time_format)
    logger = logging.getLogger()
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


class AverageMeterGroup:
    """Average meter group for multiple average meters"""

    def __init__(self):
        self.meters = OrderedDict()

    def update(self, data, n=1):
        for k, v in data.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter(k, ":4f")
            self.meters[k].update(v, n=n)

    def __getattr__(self, item):
        return self.meters[item]

    def __getitem__(self, item):
        return self.meters[item]

    def __str__(self):
        return "  ".join(str(v) for v in self.meters.values())

    def summary(self):
        return "  ".join(v.summary() for v in self.meters.values())


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        """
        Initialization of AverageMeter
        Parameters
        ----------
        name : str
            Name to display.
        fmt : str
            Format string to print the values.
        """
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = "{name}: {avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


def accuracy_mse(predict, target, scale=100.0):
    predict = predict.detach()
    target = target
    return F.mse_loss(predict, target)


def train_mlp(epoch, model, loss_func, data_loader, lr_scheduler, optimizer, device):
    epoch_loss = 0
    model.train()
    target = []
    predict = []
    lr = optimizer.param_groups[0]["lr"]
    for iter, (data, label) in enumerate(data_loader):
        data = data.float().to(device).unsqueeze(0)
        label = label.float().to(device)
        prediction = model(data)
        loss = loss_func(prediction.squeeze(), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.cpu().detach().item()
        predict.extend(prediction.squeeze().cpu().detach().numpy().tolist())
        target.extend(label.squeeze().cpu().detach().numpy().tolist())
    lr_scheduler.step()
    epoch_loss /= iter + 1
    target = np.array(target)
    predict = np.array(predict)
    return epoch_loss, target, predict


def eval_mlp(epoch, model, loss_func, data_loader, device, isprint):
    test_loss = 0
    model.eval()
    target = []
    predict = []
    with torch.no_grad():
        for iter, (data, label) in enumerate(data_loader):
            label = label.float().to(device)
            data = data.float().to(device).unsqueeze(0)
            prediction = model(data)
            loss = loss_func(prediction.squeeze(), label)
            test_loss += loss.detach().item()
            predict.extend(prediction.squeeze().cpu().numpy().tolist())
            target.extend(label.squeeze().cpu().numpy().tolist())
        test_loss /= iter + 1
    target = np.array(target)
    predict = np.array(predict)
    if isprint:
        print(f"Epoch {epoch}\ttest set loss {test_loss:.5f}\n")
    return test_loss, target, predict


def vec_to_pairwise_prob(vec):
    s_ij = vec - vec.unsqueeze(1)
    p_ij = 1 / (torch.exp(s_ij) + 1)
    return torch.triu(p_ij, diagonal=1)


class RankNetLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, labels):
        preds_prob = vec_to_pairwise_prob(preds)
        labels_prob = torch.triu((labels.unsqueeze(1) > labels).float(), diagonal=1)
        return torch.nn.functional.binary_cross_entropy(preds_prob, labels_prob)


class LambdaRankLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def lamdbaRank_scheme(self, G, D, *args):
        return torch.abs(
            torch.pow(D[:, :, None], -1.0) - torch.pow(D[:, None, :], -1.0)
        ) * torch.abs(G[:, :, None] - G[:, None, :])

    def forward(self, preds, labels, k=None, eps=1e-10, mu=10.0, sigma=1.0, device=None):
        if device is None:
            if torch.cuda.device_count():
                device = "cuda:0"
            else:
                device = "cpu"
        preds = preds[None, :]
        labels = labels[None, :]
        y_pred = preds.clone()
        y_true = labels.clone()

        y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
        y_true_sorted, _ = y_true.sort(descending=True, dim=-1)

        true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
        true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
        padded_pairs_mask = torch.isfinite(true_diffs)

        padded_pairs_mask = padded_pairs_mask & (true_diffs > 0)
        ndcg_at_k_mask = torch.zeros(
            (y_pred.shape[1], y_pred.shape[1]), dtype=torch.bool, device=device
        )
        ndcg_at_k_mask[:k, :k] = 1

        true_sorted_by_preds.clamp_(min=0.0)
        y_true_sorted.clamp_(min=0.0)

        pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
        D = torch.log2(1.0 + pos_idxs.float())[None, :]
        maxDCGs = torch.sum(((torch.pow(2, y_true_sorted) - 1) / D)[:, :k], dim=-1).clamp(min=eps)
        G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]

        weights = self.lamdbaRank_scheme(G, D, mu, true_sorted_by_preds)

        scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :]).clamp(
            min=-1e8, max=1e8
        )
        scores_diffs[torch.isnan(scores_diffs)] = 0.0
        weighted_probas = (torch.sigmoid(sigma * scores_diffs).clamp(min=eps) ** weights).clamp(
            min=eps
        )
        losses = torch.log2(weighted_probas)
        masked_losses = losses[padded_pairs_mask & ndcg_at_k_mask]
        loss = -torch.sum(masked_losses)
        return loss
