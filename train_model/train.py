import json
from argparse import ArgumentParser
from dataset import PredictDataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tvm.autotvm.tuner.model import Encoder, LSTM, MLP
from utils import (
    eval_mlp,
    train_mlp,
    RankNetLoss,
    LambdaRankLoss,
    get_logger,
    reset_seed,
    visualize_scatterplot,
    visualize_line_plot,
    visualize_line_plot2,
)
from scipy.stats import kendalltau, spearmanr
import warnings

warnings.filterwarnings("ignore")


def make_save_dir(args):
    args.save_dir = f"{args.save_dir}/{args.layout}/{args.batch}/{args.lr}/{args.wd}/{args.layers}"
    return args


def lable_weight(labels, split=1000):
    # label을 받아 bin이 100인 histogram 을 만들어
    # 해당값의 역수를 기준으로 weight를 반환
    weight = []
    dight = np.digitize(labels, np.histogram(labels, bins=split)[1])
    y = np.bincount(dight)
    ii2 = np.nonzero(y)[0]
    out = np.vstack((ii2, y[ii2]))
    lut = {i: v for i, v in zip(out[0], out[1])}
    for i in dight:
        weight.append(1 / lut[i])
    return weight


def main():
    parser = ArgumentParser()
    parser.add_argument("--linear_hidden", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_batch_size", default=4096, type=int)
    parser.add_argument("--eval_batch_size", default=4096, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    parser.add_argument("--lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("--wd", "--weight_decay", default=2e-4, type=float)
    parser.add_argument("--train_print_freq", default=1000, type=int)
    parser.add_argument("--eval_print_freq", default=1000, type=int)
    parser.add_argument("--split", default=100, type=int)
    parser.add_argument("--loss", default="mse", type=str)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--layout", default="NCHW", type=str)
    parser.add_argument("--model", default="transformer", type=str)
    parser.add_argument("--save_dir", default="result", type=str)
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--reload", default=False, action="store_true")
    parser.add_argument("--visualize", default=True, action="store_true")

    args = parser.parse_args()
    logger = get_logger()
    device = torch.device("cuda")
    n_head = 4
    reset_seed(args.seed)
    dataset = PredictDataset(
        folder_path=f"{args.dataset_dir}/{args.layout}/{args.batch}",
        layout=args.layout,
        cache=args.reload,
    )

    len_dataset = len(dataset)
    train_set, test_set = torch.utils.data.random_split(
        dataset, [round(len_dataset * 0.9), len_dataset - round(len_dataset * 0.9)]
    )
    args.mean = float(dataset.MEAN)
    args.std = float(dataset.STD)
    args.train_length = int(len(dataset) * 0.9)
    args.test_length = int(len(dataset) * 0.1)
    args.data_length = int(len(dataset))
    train_weight = torch.tensor(lable_weight(train_set.dataset.labels, args.split))
    test_weight = torch.tensor(lable_weight(test_set.dataset.labels, args.split))
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=train_weight, num_samples=len(train_weight), replacement=True
    )
    test_sampler = torch.utils.data.WeightedRandomSampler(
        weights=test_weight, num_samples=len(test_weight), replacement=True
    )
    args_dict = dict(vars(args))
    for i, ii in args_dict.items():
        print(i, ii)
    if args.model.lower() == "transformer":
        model = Encoder(
            embed_size=748,
            num_heads=n_head,
            num_layers=args.layers,
            hidden_size=args.linear_hidden,
            dropout=args.dropout,
        ).to(device)
    elif args.model.lower() == "lstm":
        model = LSTM(748, 1024, 1, 3).to(device)
    elif args.model.lower() == "mlp":
        model = MLP(748, 1024, 1).to(device)
    data_loader = DataLoader(
        train_set.dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
    )
    test_data_loader = DataLoader(
        test_set.dataset,
        sampler=test_sampler,
        batch_size=args.eval_batch_size,
        num_workers=1,
        pin_memory=False,
        drop_last=True,
    )

    print(model)
    ## select loss
    if args.loss.lower() == "mse":
        criterion = nn.MSELoss().to(device)
    elif args.loss.lower() == "l1":
        criterion = nn.L1Loss().to(device)
    elif args.loss.lower() == "rank":
        criterion = RankNetLoss().to(device)
    elif args.loss.lower() == "lambdarank":
        criterion = LambdaRankLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    args = make_save_dir(args)
    best_loss = 10**10
    train_loss_list = []
    test_loss_list = []
    train_rank_list1 = []
    train_rank_list2 = []
    test_rank_list1 = []
    test_rank_list2 = []

    os.makedirs(args.save_dir, exist_ok=True)
    for epoch in range(args.epochs):
        train_loss, train_target, train_predict = train_mlp(
            epoch=epoch,
            model=model,
            loss_func=criterion,
            data_loader=data_loader,
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            device=device,
        )
        eval_loss, eval_target, eval_predict = eval_mlp(
            epoch=epoch,
            model=model,
            loss_func=criterion,
            data_loader=test_data_loader,
            device=device,
            isprint=False,
        )
        train_loss_list.append(train_loss)
        test_loss_list.append(eval_loss)
        train_rank1 = kendalltau(train_predict, train_target)[0]
        train_rank2 = spearmanr(train_predict, train_target)[0]
        logger.info(
            f"[Train][{epoch+1}/{args.epochs}] Loss: {train_loss:.6f}\tKendalltau: {train_rank1:.6f}\tSpearmanr :{train_rank2:.6f}"
        )
        eval_rank1 = kendalltau(eval_predict, eval_target)[0]
        eval_rank2 = spearmanr(eval_predict, eval_target)[0]

        logger.info(
            f"[Test][{epoch+1}/{args.epochs}] Loss: {train_loss:.6f}\tKendalltau: {eval_rank1:.6f}\tSpearmanr :{eval_rank2:.6f}"
        )
        train_rank_list1.append(train_rank1)
        test_rank_list1.append(eval_rank1)
        train_rank_list2.append(train_rank2)
        test_rank_list2.append(eval_rank2)
        if best_loss > eval_loss:
            best_loss = eval_loss
            torch.save(model.state_dict(), f"{args.save_dir}/model_best.pth")
            visualize_scatterplot(args, "train_best", train_predict, train_target)
            visualize_scatterplot(args, "test_best", eval_predict, eval_target)
        if epoch + 1 % 10 == 0:
            torch.save(model.state_dict(), f"{args.save_dir}/model_{epoch}.pth")
            visualize_scatterplot(args, f"train_{epoch}", train_predict, train_target)
            visualize_scatterplot(args, f"test_{epoch}", eval_predict, eval_target)

    torch.save(model.state_dict(), f"{args.save_dir}/model_last.pth")

    args.Tkendalltau = kendalltau(train_predict, train_target)[0]
    args.Ekendalltau = kendalltau(eval_predict, eval_target)[0]
    args.TSpearmanr = spearmanr(train_predict, train_target)[0]
    args.ESpearmanr = spearmanr(eval_predict, eval_target)[0]
    with open(os.path.join(f"{args.save_dir}", "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    if args.visualize:
        visualize_scatterplot(args, "train", train_predict, train_target)
        visualize_scatterplot(args, "test", eval_predict, eval_target)
        visualize_line_plot2(args, train_rank_list1, train_rank_list2, "Correlation in Train")
        visualize_line_plot2(args, test_rank_list1, test_rank_list2, "Correlation in Test")
        visualize_line_plot(args, train_loss_list, test_loss_list, "Loss")
        visualize_line_plot(args, train_loss_list[10:], test_loss_list[10:], "Loss(edit)")

    np.save(f"{args.save_dir}/train_loss.npy", train_loss_list)
    np.save(f"{args.save_dir}/test_loss.npy", test_loss_list)
    np.save(f"{args.save_dir}/test_target.npy", eval_target)
    np.save(f"{args.save_dir}/test_predict.npy", eval_predict)
    np.save(f"{args.save_dir}/train_target.npy", train_target)
    np.save(f"{args.save_dir}/train_predict.npy", train_predict)


if __name__ == "__main__":
    main()
