"""
Author: Jayanth Raman

Source: pytorch/examples
"""
import argparse
import logging
import os
import random
import sys
import time
import torch
import torch.distributed
import torchvision


logger = logging.getLogger(__name__)


class SyntheticImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        nclasses: int = 10,
        nsamples: int = 100,
        dimvalmin=(3, 200, 200),
        dimvalmax=(3, 500, 500),
        transform=None,
        target_transform=None,
    ):
        assert len(dimvalmin) == len(dimvalmax) == 3
        self.nclasses = nclasses
        self.nsamples = nsamples
        self.dimmin = dimvalmin
        self.dimmax = dimvalmax
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx: int, lo=0, hi=255):
        """Values are in the range [lo, hi] -- inclusive at both ends"""
        assert -1 < idx < self.nsamples
        dim = []
        for d1, d2 in zip(self.dimmin, self.dimmax):
            dim.append(random.randint(d1, d2))
        img = torch.randint(low=lo, high=hi + 1, size=dim, dtype=torch.uint8)
        lab = random.randint(0, self.nclasses - 1)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            lab = self.target_transform(lab)
        return img, lab


def parse_args():
    model_names = sorted(
        name
        for name in torchvision.models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(torchvision.models.__dict__[name])
    )

    parser = argparse.ArgumentParser(
        description="PyTorch Distributed Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-a",
        "--arch",
        metavar="ARCH",
        default="resnet18",
        choices=model_names,
        help="model architecture: " + " | ".join(model_names),
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--epochs",
        default=2,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size, this is the total "
        "batch size of all GPUs on the current node when "
        "using Data Parallel or Distributed Data Parallel."
        "Actual batch-size in DDP is this / num-GPU.",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (every N batches)",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        action="store_true",
        help="use pre-trained model",
    )
    parser.add_argument(
        "--world-size",
        default=-1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=-1, type=int, help="node rank for distributed-node training"
    )
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:23456",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--backend",
        "--dist-backend",
        default="nccl",
        choices=("gloo", "mpi", "nccl"),
        help="distributed backend",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--gpu",
        default=None,
        type=int,
        help="GPU id to use."
        "Turns off distributed processing and limit processing to a single GPU.",
    )
    parser.add_argument(
        "--multiprocessing-distributed",
        "--multiprocessing",
        "--mp",
        action="store_true",
        help="Use multi-processing distributed training to launch "
        "N processes per node, which has N GPUs. This is the "
        "fastest way to use PyTorch for either single node or "
        "multi node data parallel training",
    )
    parser.add_argument(
        "--nsamples",
        "-n",
        default=1000,
        type=int,
        help="Number of samples in the synthetic dataset.",
    )
    parser.add_argument(
        "--pin-memory", action="store_true", help="pin-memory for data-loader"
    )
    return parser.parse_args()


def main():
    format = (
        "%(asctime)s.%(msecs)03d %(levelname)s %(funcName)s:L%(lineno)d %(message)s"
    )
    logging.basicConfig(
        stream=sys.stdout, level=logging.DEBUG, format=format, datefmt="%H:%M:%S"
    )
    logger.info("sys.argv: %s", sys.argv)
    logger.info("cmd: %s", " ".join(sys.argv))
    args = parse_args()
    logger.info("args: %s", args)
    ngpus = torch.cuda.device_count()  # GPUs per node
    logger.info("Num GPUs: %s", ngpus)
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    logger.info("Distributed training on %s nodes", args.world_size)
    logger.info("NCCL available: %s", torch.distributed.is_nccl_available())
    if args.multiprocessing_distributed:
        logger.info("Multiprocessing distributed training is ON")
        args.world_size = ngpus * args.world_size
        torch.multiprocessing.spawn(worker, nprocs=ngpus, args=(ngpus, args))
    else:
        logger.info("Multiprocessing distributed training is OFF.")
        worker(args.gpu, ngpus, args)
    logger.info("")


def worker(gpu, ngpus, args):
    format = f"[w{gpu}] %(asctime)s.%(msecs)03d %(levelname)s %(funcName)s:L%(lineno)d %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=format, datefmt="%H:%M:%S")
    logger.info("gpu: %s, ngpus: %s, args: %s", gpu, ngpus, args)
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus + gpu
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "23456"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # OFF, INFO, DETAIL
        torch.distributed.init_process_group(
            backend=args.backend,  # gloo, mpi, nccl
            # init_method=args.dist_url,
            init_method="env://",
            world_size=args.world_size,
            rank=args.rank,
        )
        logger.info("rank: %s", torch.distributed.get_rank())
    logger.info("model arch=%s, pretrained=%s", args.arch, args.pretrained)
    model = torchvision.models.__dict__[args.arch](pretrained=args.pretrained)

    if not torch.cuda.is_available():
        logger.warning("No GPU -- using (slow) CPU.")
    elif args.distributed:
        if gpu is not None:
            torch.cuda.set_device(gpu)
            model.cuda(gpu)
            args.batch_size = int(args.batch_size / ngpus)
            logger.info("New batch size: %s", args.batch_size)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        if args.arch.startswith("alexnet") or args.arch.startswith("vgg"):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    criterion = torch.nn.CrossEntropyLoss().cuda(gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # torch.backends.cudnn.benchmark = True

    MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomAffine(20),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(MEAN, STD),
        ]
    )
    ds = SyntheticImageDataset(
        nsamples=args.nsamples,
        transform=transform,
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(ds)
    else:
        train_sampler = None
    logger.info("Sampler: %s", train_sampler)

    dataloader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        sampler=train_sampler,
    )

    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train(gpu, dataloader, model, criterion, optimizer, epoch, args.print_freq)


def train(gpu, loader, model, criterion, optimizer, epoch, print_freq):
    model.train()
    for i, (images, target) in enumerate(loader):
        start = time.time()
        if gpu is not None:
            images = images.cuda(gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(gpu, non_blocking=True)
        #
        output = model(images)
        loss = criterion(output, target)
        #
        # skip accuracy calculations
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        if i % print_freq == 0:
            logger.debug(f"shapes: images: {images.shape}, target: {target.shape}")
            logger.info(
                f"epoch {epoch}: batch {i}: gpu: {gpu}, batch time: {time.time() - start}"
            )


if __name__ == "__main__":
    main()
