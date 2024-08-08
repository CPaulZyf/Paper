#!/usr/bin/env python
import torch
import argparse
import os
import time
import warnings
import numpy as np
import logging
import torchvision
from flcore.servers.serverfedapp import FedAPP
from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)


def run(args):
    time_list = []
    reporter = MemReporter()

    args.models = [
        # 'FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)',
        'torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes)',
        'mobilenet_v2(pretrained=False, num_classes=args.num_classes)',
        'torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes)',
        'torchvision.models.shufflenet_v2_x1_0(pretrained=False, num_classes=args.num_classes)',
        'torchvision.models.efficientnet_b0(pretrained=False, num_classes=args.num_classes)'
    ]

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        for model in args.models:
            print(model)

        server = FedAPP(args, i)
        server.train()
        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="1")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10", help="[FMNIST, Cifar10, Cifar100]")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model_family", type=str, default="HtFE5")
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.01,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=500)
    parser.add_argument('-ls', "--local_epochs", type=int, default=1,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAPP")
    parser.add_argument('-patience', "--patience", type=int, default=20,
                        help="stop rounds")
    parser.add_argument('-mr', "--min_round", type=int, default=50,
                        help="stop rounds")

    parser.add_argument('-nc', "--num_clients", type=int, default=30,
                        help="Total number of clients")
    parser.add_argument('-ncp', "--num_clients_part", type=int, default=5,
                        help="Total number of clients")
    parser.add_argument('-non_iid', "--non_iid", type=str, default='dir',
                        help="non-iid[dir, pat]")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=5,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=2,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='temp')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-ml', "--max_len", type=int, default=200)

    parser.add_argument('-dt', "--delay_time", type=int, default=5,
                        help="delay_time")
    parser.add_argument('-dg', "--delay_group", type=int, default=5,
                        help="delay_time")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0)
    args = parser.parse_args()
    if args.dataset == "Cifar100":
        args.num_classes = 100

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

    print("=" * 50)

    print("Algorithm: {}".format(args.algorithm))
    print("Local batch size: {}".format(args.batch_size))
    print("Local steps: {}".format(args.local_epochs))
    print("Local learing rate: {}".format(args.local_learning_rate))
    print("Local learing rate decay: {}".format(args.learning_rate_decay))
    if args.learning_rate_decay:
        print("Local learing rate decay gamma: {}".format(args.learning_rate_decay_gamma))
    print("Total number of clients: {}".format(args.num_clients))
    print("Running times: {}".format(args.times))
    print("Dataset: {}".format(args.dataset))
    print("Non-i.i.d: {}".format(args.non_iid))
    print("Number of classes: {}".format(args.num_classes))
    print("Backbone: {}".format(args.model_family))
    print("Using device: {}".format(args.device))
    print("Auto break: {}".format(args.auto_break))
    if not args.auto_break:
        print("Global rounds: {}".format(args.global_rounds))
    print("=" * 50)
    run(args)

