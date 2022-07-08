from __future__ import print_function

import argparse
import os
import random

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import models

from dataloaders import dataloader_cifar as dataloader
from models import bit_models
from models.PreResNet import *
from models.resnet import SupCEResNet
from train_cifar import run_train_loop

from codivide_utils import codivide_gmm, codivide_ccgmm
from utils import load_net_optimizer_from_ckpt_to_device, get_epoch_from_checkpoint

import csv


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR Training")
    parser.add_argument("--batch_size", default=64, type=int, help="train batchsize")
    parser.add_argument(
        "--lr",
        "--learning_rate",
        default=0.02,
        type=float,
        help="initial learning rate",
    )
    parser.add_argument("--noise_mode", default="sym")
    parser.add_argument("--alpha", default=4.0, type=float, help="parameter for Beta")
    parser.add_argument(
        "--alpha-loss", default=0.5, type=float, help="parameter for Beta in loss"
    )
    parser.add_argument(
        "--lambda_u", default=25, type=float, help="weight for unsupervised loss"
    )
    parser.add_argument(
        "--p_threshold", default=0.5, type=float, help="clean probability threshold"
    )
    parser.add_argument("--T", default=0.5, type=float, help="sharpening temperature")
    parser.add_argument("--num_epochs", default=360, type=int)
    parser.add_argument("--r", default=0.5, type=float, help="noise ratio")
    parser.add_argument("--id", default="")
    parser.add_argument("--seed", default=123)
    parser.add_argument("--gpuid", default=0, type=int)
    parser.add_argument(
        "--data_path", default="./cifar-10", type=str, help="path to dataset"
    )
    parser.add_argument("--net", default="resnet18", type=str, help="net")
    parser.add_argument("--method", default="reg", type=str, help="method")
    parser.add_argument("--dataset", default="cifar10", type=str)
    parser.add_argument("--experiment-name", required=True, type=str)
    parser.add_argument(
        "--aug", dest="aug", action="store_true", help="use stronger aug"
    )
    parser.add_argument(
        "--use-std", dest="use_std", action="store_true", help="use stronger aug"
    )
    parser.add_argument("--drop", dest="drop", action="store_true", help="use drop")
    parser.add_argument(
        "--not-rampup", dest="not_rampup", action="store_true", help="not rumpup"
    )
    parser.add_argument(
        "--supcon", dest="supcon", action="store_true", help="use supcon"
    )
    parser.add_argument(
        "--use-aa", dest="use_aa", action="store_true", help="use supcon"
    )
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument(
        "--window_mode",
        choices=["mean", "exp_smooth"],
        default="mean",
        help="method for the computation of the weights",
    )
    parser.add_argument("--lambda_w_eps", default=0.1, type=float)
    parser.add_argument(
        "--weight_mode",
        choices=["f1_score", "acc"],
        default="f1_score",
        help="method for the computation of the weights",
    )
    parser.add_argument("--weightsLu", dest="weightsLu", action="store_true")
    parser.add_argument("--no-weightsLu", dest="weightsLu", action="store_false")
    parser.set_defaults(weightsLu=True)
    parser.add_argument("--weightsLr", dest="weightsLr", action="store_true")
    parser.add_argument("--no-weightsLr", dest="weightsLr", action="store_false")
    parser.set_defaults(weightsLr=True)
    parser.add_argument(
        "--class-conditional", default=False, dest="ccgmm", action="store_true"
    )
    parser.set_defaults(ccgmm=False)
    parser.add_argument(
        "--skip-warmup", default=False, dest="skip_warmup", action="store_true"
    )
    parser.set_defaults(skip_warmup=False)
    parser.add_argument(
        "--num_workers",
        default=5,
        type=int,
        help="num of dataloader workers. Colab recommends 2.",
    )
    parser.add_argument(
        "--root", default=".", type=str, help="root of the checkpoint dir"
    )
    parser.add_argument(
        "--resume", default=None, type=str, help="path of the model to load"
    )
    parser.add_argument(
        "--save-models", default=False, dest="save_models", action="store_true"
    )
    parser.set_defaults(save_models=False)
    parser.add_argument(
        "--codivide-log", default=False, dest="enableLog", action="store_true"
    )
    parser.set_defaults(enableLog=False)
    parser.add_argument("--lambda-x", default=0, dest="lambda_x", type=float)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.device(args.gpuid)
        torch.cuda.manual_seed_all(args.seed)
        args.device = f"cuda:{args.gpuid}"
    else:
        args.device = "cpu"

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args


def linear_rampup(current, warm_up, lambda_u, rampup_length=16):
    current = np.clip((current - warm_up) / rampup_length, 0.0, 1.0)
    return lambda_u * float(current)


class SemiLoss(object):
    def __call__(
        self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up, lambda_u
    ):
        probs_u = torch.softmax(outputs_u, dim=1)

        # Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        # Lu = torch.mean((probs_u - targets_u) ** 2)
        Lx = torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1)
        Lu = (probs_u - targets_u) ** 2

        return Lx, Lu, linear_rampup(epoch, warm_up, lambda_u)


class SemiLoss_uncertainty(object):
    def __call__(
        self,
        outputs_x,
        targets_x,
        uncertainty_weights_x,
        outputs_u,
        targets_u,
        uncertainty_weights_u,
        epoch,
        warm_up,
        lambda_u,
    ):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(
            uncertainty_weights_x
            * torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1)
        )
        Lu = torch.mean(
            uncertainty_weights_u * torch.mean((probs_u - targets_u) ** 2, dim=1)
        )

        return Lx, Lu, linear_rampup(epoch, warm_up, lambda_u)


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


def create_model_reg(
    net="resnet18",
    dataset="cifar100",
    num_classes=100,
    device="cuda:0",
    drop=0,
    root=".",
):
    if net == "resnet18":
        model = ResNet18(num_classes=num_classes, drop=drop)
        model = model.to(device)
        return model
    else:
        model = SupCEResNet(net, num_classes=num_classes)
        model = model.to(device)
        return model


def create_model_selfsup(
    net="resnet18",
    dataset="cifar100",
    num_classes=100,
    device="cuda:0",
    drop=0,
    root=".",
):
    chekpoint = torch.load("{}/pretrained/ckpt_{}_{}.pth".format(root, dataset, net))
    sd = {}
    for ke in chekpoint["model"]:
        nk = ke.replace("module.", "")
        sd[nk] = chekpoint["model"][ke]
    model = SupCEResNet(net, num_classes=num_classes)
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    return model


def create_model_bit(
    net="resnet18",
    dataset="cifar100",
    num_classes=100,
    device="cuda:0",
    drop=0,
    root=".",
):
    if net == "resnet50":
        model = bit_models.KNOWN_MODELS["BiT-S-R50x1"](
            head_size=num_classes, zero_head=True
        )
        model.load_from(np.load(f"{root}/pretrained/BiT-S-R50x1.npz"))
        model = model.to(device)
    elif net == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(512 * 1, num_classes)
        model = model.to(device)
    else:
        raise ValueError()
    return model


def main():
    args = parse_args()

    weightsString = ""

    if args.weightsLu:
        print("Using weights in Lu")
    else:
        print("No weights in Lu")
        weightsString = weightsString + "_NOweightsLu"
    if args.weightsLr:
        print("Using weights in Lr")
    else:
        print("No weights in Lr")
        weightsString = weightsString + "_NOweightsLr"

    checkpoint_root = f"{args.root}/checkpoint"
    os.makedirs(checkpoint_root, exist_ok=True)

    experiment_prefix = f"""{args.experiment_name}_{args.dataset}_{args.r:.2f}_{args.lambda_u:.1f}_{args.noise_mode}{weightsString}"""
    experiment_folder = f"""{checkpoint_root}/{experiment_prefix}"""
    detailed_losses_folder = f"""{experiment_folder}/detailedLosses"""
    figures_folder = f"""{experiment_folder}/codivide_figures"""
    os.makedirs(detailed_losses_folder, exist_ok=True)
    os.makedirs(figures_folder, exist_ok=True)

    detailed_losses_file = f"""{detailed_losses_folder}/{experiment_prefix}_losses_per_class_epoch_{{}}.txt"""

    model_checkpoint_folder = None
    if args.save_models:
        model_checkpoint_folder = f"""{experiment_folder}/models"""
        print(f"Models will be saved every 5 epochs at {model_checkpoint_folder}")
        os.makedirs(model_checkpoint_folder, exist_ok=True)

    stats_log = open(f"{experiment_folder}/{experiment_prefix}_stats.txt", "a")
    test_log = open(f"{experiment_folder}/{experiment_prefix}_acc.txt", "a")
    loss_log = open(f"{experiment_folder}/{experiment_prefix}_loss.txt", "a")
    codivide_log = open(f"{experiment_folder}/{experiment_prefix}_codivide.txt", "a")

    # define co-divide policy
    if args.ccgmm:
        codivide_policy = codivide_ccgmm
    else:
        codivide_policy = codivide_gmm

    # define warmup
    if args.dataset == "cifar10":
        if args.method == "reg":
            warm_up = 20 if args.aug else 10
        else:
            warm_up = 5
        num_classes = 10
    elif args.dataset == "cifar100":
        if args.method == "reg":
            warm_up = 60 if args.aug else 30
        else:
            warm_up = 5
        num_classes = 100
    else:
        raise ValueError("Wrong dataset")

    if args.skip_warmup:
        warm_up = 0
        print("WARNING! Skipping warm up for debugging purposes")
    print("Warm up epochs = ", warm_up)

    weights_log = open(f"{experiment_folder}/{experiment_prefix}_weights.txt", "w")
    w_fields = (
        ["epoch"]
        + [f"w_net_1_{cls}" for cls in range(num_classes)]
        + [f"w_net_2_{cls}" for cls in range(num_classes)]
    )

    # creating a csv writer object
    csvwriter = csv.writer(weights_log)
    # writing the fields
    csvwriter.writerow(w_fields)
    weights_log.flush()

    training_losses_log = open(
        f"{experiment_folder}/{experiment_prefix}_training_losses.txt", "w"
    )
    tl_fields = ["epoch", "L_x", "L_u", "lambda_u", "L_reg", "L_total"]

    # creating a csv writer object
    csvwriter = csv.writer(training_losses_log)
    # writing the fields
    csvwriter.writerow(tl_fields)
    training_losses_log.flush()

    loader = dataloader.cifar_dataloader(
        args.dataset,
        r=args.r,
        noise_mode=args.noise_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        root_dir=args.data_path,
        log=stats_log,
        noise_file="%s/%.2f_%s.json" % (args.data_path, args.r, args.noise_mode),
        stronger_aug=args.aug,
    )

    print("| Building net")
    if args.method == "bit":
        create_model = create_model_bit
    elif args.method == "reg":
        create_model = create_model_reg
    elif args.method == "selfsup":
        create_model = create_model_selfsup
    else:
        raise ValueError()
    net1 = create_model(
        net=args.net,
        dataset=args.dataset,
        num_classes=num_classes,
        device=args.device,
        drop=args.drop,
        root=args.root,
    )
    net2 = create_model(
        net=args.net,
        dataset=args.dataset,
        num_classes=num_classes,
        device=args.device,
        drop=args.drop,
        root=args.root,
    )
    cudnn.benchmark = False  # True

    criterion = SemiLoss()

    if args.resume is None:
        optimizer1 = optim.SGD(
            net1.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
        )
        optimizer2 = optim.SGD(
            net2.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4
        )
        resume_epoch = 1  # TODO: Ask Ricardo if it should be 0 or 1.

    else:
        net1, optimizer1 = load_net_optimizer_from_ckpt_to_device(
            net1, args, f"{args.resume}_1.pt", args.device
        )
        net2, optimizer2 = load_net_optimizer_from_ckpt_to_device(
            net2, args, f"{args.resume}_2.pt", args.device
        )
        resume_epoch = get_epoch_from_checkpoint(args.resume)

    sched1 = torch.optim.lr_scheduler.StepLR(optimizer1, 150, gamma=0.1)
    sched2 = torch.optim.lr_scheduler.StepLR(optimizer2, 150, gamma=0.1)

    CE = nn.CrossEntropyLoss(reduction="none")
    CEloss = nn.CrossEntropyLoss()
    if args.noise_mode == "asym":
        conf_penalty = NegEntropy()
    else:
        conf_penalty = None
    all_loss = [[], []]  # save the history of losses from two networks

    run_train_loop(
        net1,
        optimizer1,
        sched1,
        net2,
        optimizer2,
        sched2,
        criterion,
        CEloss,
        CE,
        loader,
        args.p_threshold,
        warm_up,
        args.num_epochs,
        all_loss,
        args.batch_size,
        num_classes,
        args.device,
        args.lambda_u,
        args.T,
        args.alpha,
        args.noise_mode,
        args.dataset,
        args.r,
        conf_penalty,
        stats_log,
        loss_log,
        test_log,
        weights_log,
        training_losses_log,
        detailed_losses_file,
        args.window_size,
        args.window_mode,
        args.lambda_w_eps,
        args.weight_mode,
        args.experiment_name,
        args.weightsLu,
        args.weightsLr,
        args.enableLog,
        figures_folder,
        codivide_policy,
        codivide_log,
        model_checkpoint_folder,
        resume_epoch,
    )


if __name__ == "__main__":
    main()
