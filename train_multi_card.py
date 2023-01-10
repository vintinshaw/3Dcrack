import argparse
import time
import torch
import torchvision
from torch import distributed as dist
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--load-from', help='the checkpoint file to load weights from')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
             'not be supported in version v0.22.0. Override some settings in the '
             'used config, the key-value pair in xxx=yyy format will be merged '
             'into config file. If the value to be overwritten is a list, it '
             'should be like key="[a,b]" or key=a,b It also allows nested '
             'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
             'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def reduce_loss(tensor, rank, world_size):
    with torch.no_grad():
        dist.reduce(tensor, dst=0)
        if rank == 0:
            tensor /= world_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, help="local gpu id")
    args = parser.parse_args()

    batch_size = 128
    epochs = 5
    lr = 0.001

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    global_rank = dist.get_rank()

    net = resnet18()
    net.cuda()
    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank)

    data_root = 'dataset'
    trainset = MNIST(root=data_root,
                     download=True,
                     train=True,
                     transform=ToTensor())

    valset = MNIST(root=data_root,
                   download=True,
                   train=False,
                   transform=ToTensor())

    sampler = DistributedSampler(trainset)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              shuffle=False,
                              pin_memory=True,
                              sampler=sampler)

    val_loader = DataLoader(valset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    net.train()
    for e in range(epochs):
        # DistributedSampler deterministically shuffle data
        # by seting random seed be current number epoch
        # so if do not call set_epoch when start of one epoch
        # the order of shuffled data will be always same
        sampler.set_epoch(e)
        for idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.cuda()
            labels = labels.cuda()
            output = net(imgs)
            loss = criterion(output, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            reduce_loss(loss, global_rank, world_size)
            if idx % 10 == 0 and global_rank == 0:
                print('Epoch: {} step: {} loss: {}'.format(e, idx, loss.item()))
    net.eval()
    with torch.no_grad():
        cnt = 0
        total = len(val_loader.dataset)
        for imgs, labels in val_loader:
            imgs, labels = imgs.cuda(), labels.cuda()
            output = net(imgs)
            predict = torch.argmax(output, dim=1)
            cnt += (predict == labels).sum().item()

    if global_rank == 0:
        print('eval accuracy: {}'.format(cnt / total))


if __name__ == '__main__':
    main()
