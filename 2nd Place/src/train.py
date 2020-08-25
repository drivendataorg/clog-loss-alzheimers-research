import argparse
from pathlib import Path
import random
import os
import itertools as it

import apex
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import tqdm
import pickle

from model import get_model
from metric import Accuracy, MatthewsCorrcoef
from dataset import CloudsDS, CloudsDSPretrain, train_transform, dev_transform, collate_fn, collate_fn_3d
from utils import get_data_groups


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data')
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--lmdb', type=str, required=True)

    parser.add_argument('--work-dir', default='', type=str,
                        help='Working directory')
    parser.add_argument('--load', default='', type=str,
                        help='Load model (default: none)')
    parser.add_argument('--resume', default='', type=str,
                        help='Path to latest checkpoint (default: none)')
    parser.add_argument('--model', type=str, default='lstm')
    parser.add_argument('--encoder', type=str, default='efficientnet-b0')
    parser.add_argument('--fps', type=int, default=2)
    parser.add_argument('--size', type=int, default=160)
    parser.add_argument('--in-channels', type=int, default=1)
    parser.add_argument('--n-classes', type=int, default=1)
    parser.add_argument('--mixup', type=float, default=-1)
    parser.add_argument('--batch-accum', type=int, default=1)
    
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--fold', type=int, default=0)

    parser.add_argument('--pl', type=str, default=None)
    parser.add_argument('--pretrain', type=str, default=None)
    parser.add_argument('--pretrain2', type=str, default=None)
    parser.add_argument('--ft', action='store_true')

    parser.add_argument('--teachers', type=str, default=None)
#     parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--temperature', type=float, default=20)

    parser.add_argument('--workers', '-j', type=int, default=8, required=False)

    parser.add_argument('--epochs', '-e', type=int, default=300)
    parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='Batch size per process (default: 8)')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-4,
                        metavar='LR',
                        help='Initial learning rate. Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256. A warmup schedule will also be applied over the first 5 epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    
    parser.add_argument('--hyper-lr', type=float, default=1e-8, help='beta, only applicable for HD')
    parser.add_argument('--alpha', type=float, default=1e-6)
    parser.add_argument('--grad-clipping', type=float, default=100.0)
    parser.add_argument('--mu', type=float, default=0.99999)
    parser.add_argument('--first-order', action='store_true')
    
    parser.add_argument('--scheduler', type=str, default='cos')
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--T-max', type=int, default=5)

    parser.add_argument('--seed', type=int, default=314159,
                        help='Random seed')
    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--sync-bn', action='store_true',
                        help='Enabling apex sync BN.')

    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--opt-level', type=str, default='O1')
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def loss_fn_kd(outputs, labels, teacher_outputs):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    alpha = args.alpha
    T = args.temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1),
                             F.softmax(teacher_outputs / T, dim=1)) * (alpha * T * T) + \
              nn.CrossEntropyLoss()(outputs, labels) * (1. - alpha)

    return KD_loss


def epoch_step(loader, desc, model, criterion, metrics, opt=None, batch_accum=1, teachers=None):
    is_train = opt is not None
    if is_train:
        model.train()
    else:
        model.eval()
    use_mixup = (args.mixup > 0) and is_train

    pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)
    loc_loss = n = 0
    loc_accum = 1
    if teachers is not None:
        loc_loss_t = 0

    for x, y in loader:
        x = [_x.cuda(args.gpu, non_blocking=True) for _x in x] if isinstance(x, list) else x.cuda(args.gpu, non_blocking=True)
        y = y.cuda(args.gpu, non_blocking=True)
#         z = z.cuda(args.gpu, non_blocking=True)

        if use_mixup:
            x, y_a, y_b, lam = mixup_data(x, y, args.mixup)
            logits = model(x)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam) / batch_accum
        else:
            logits = model(x)
            loss = criterion(logits, y) / batch_accum
            
#         first_grad = ag.grad(loss, model.parameters(), create_graph=True, retain_graph=True)
#         hyper_opt.compute_hg(model, first_grad)

        if is_train:
            if teachers is not None and args.alpha != 1:
                alpha = args.alpha
                T = args.temperature
                logits_over_T = logits / T
                loss_t = 0
                for teacher in teachers:
                    with torch.no_grad():
                        logits_t = teacher(x)
                    loss_t += nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(logits_over_T, dim=1),
                                                                  torch.softmax(logits_t / T, dim=1))
                #                 loss_t = sum(nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(logits_over_T, dim=1), torch.softmax(teacher(x)/T, dim=1))
                #                              for teacher in teachers)
                if alpha == 0:
                    loss = loss_t / len(teachers) / batch_accum
                else:
                    loss = alpha * loss + (1 - alpha) * T * T * loss_t / len(teachers) / batch_accum

            if args.fp16:
                with apex.amp.scale_loss(loss, opt) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if loc_accum == batch_accum:
                opt.step()
                opt.zero_grad()
#                 hyper_optim.hyper_step(vg.val_grad(net))
#                 clear_grad(net)
                loc_accum = 1
            else:
                loc_accum += 1

            logits = logits.detach()

        bs = len(x)
        loc_loss += loss.item() * bs * batch_accum
        if teachers is not None:
            loc_loss_t += loss_t.item() * bs * batch_accum

        n += bs

        y_cpu_np = y.cpu().numpy()
        logits_cpu_np = torch.sigmoid(logits).cpu().numpy()

        for metric in metrics.values():
            metric.update(logits_cpu_np, y_cpu_np)

        torch.cuda.synchronize()

        if args.local_rank == 0:
            postfix = {
                'loss': f'{loc_loss / n:.3f}',
            }
            if teachers is not None:
                postfix.update({'loss_t': f'{loc_loss_t / n:.3f}'})

            postfix.update({k: f'{metric.evaluate():.3f}' for k, metric in metrics.items()})
            if is_train:
                postfix.update({'lr': next(iter(opt.param_groups))['lr']})
            pbar.set_postfix(**postfix)
            pbar.update()

    if is_train and loc_accum != batch_accum:
        opt.step()
        opt.zero_grad()

    pbar.close()
    
    return loc_loss / n


def plot_hist(history, path):
    history_len = len(history)
    n_rows = history_len // 2 + 1
    n_cols = 2
    plt.figure(figsize=(12, 4 * n_rows))
    for i, (m, vs) in enumerate(history.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        for k, v in vs.items():
            if 'loss' in m:
                ep = np.argmin(v)
            else:
                ep = np.argmax(v)
            plt.title(f'{v[ep]:.4} on {ep}')
            plt.plot(v, label=f'{k} {v[-1]:.4}')

        plt.xlabel('#epoch')
        plt.ylabel(f'{m}')
        plt.legend()
        plt.grid(ls='--')

    plt.tight_layout()
    plt.savefig(path / 'evolution.png')
    plt.close()


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    bs = x.size(0)
    index = torch.randperm(bs)
    mixed_x = lam * x + (1 - lam) * x[index, :]

    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def add_weight_decay(model, weight_decay=1e-4, skip_list=('bn',)):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= args.world_size

    return rt


# https://discuss.pytorch.org/t/how-to-concatenate-different-size-tensors-from-distributed-processes/44819
# https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/comm.py
def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = args.world_size
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    torch.distributed.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    torch.distributed.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def main():
    global args

    args = parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True
    if args.deterministic:
        set_seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 0
    args.world_size = 1
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, 'Amp requires cudnn backend to be enabled.'

    # create model
    model = get_model(args.model, args.encoder, args.n_classes)
    
    if args.sync_bn:
        print('using apex synced BN')
        model = apex.parallel.convert_syncbn_model(model)

    model.cuda()

    # Scale learning rate based on global batch size
    print(f'lr={args.lr}, opt={args.opt}')
    if args.opt == 'adam':
        opt = apex.optimizers.FusedAdam(model.parameters(),  # add_weight_decay(model, args.weight_decay, ('bn', )),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay,
                                        )
#         global hyper_opt
#         hyper_opt = MuAdam(opt, args.hyper_lr, args.grad_clipping, args.first_order, args.mu, args.alpha, torch.device('cuda'))
    elif args.opt == 'sgd':
        opt = apex.optimizers.FusedSGD(add_weight_decay(model, args.weight_decay, ('bn',)),  # model.parameters(),
                              args.lr,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay
                              )
    else:
        raise

    
    # Initialize Amp. Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    if args.fp16:
        model, opt = apex.amp.initialize(model, opt,
                                         opt_level=args.opt_level,
                                         keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                         loss_scale=args.loss_scale
                                         )

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

    bce_loss = nn.BCEWithLogitsLoss()
    if args.n_classes == 1:
        if args.loss == 'bce':
            criterion = bce_loss
        else:
            print(f'no such loss "{args.loss}"')
            raise
    else:
        criterion = nn.CrossEntropyLoss()

    history = {
        k: {k_: [] for k_ in ['train', 'dev']}
        for k in ['loss']
    }
    best_score = 0
    if args.n_classes == 1:
        metrics = {
            'score': MatthewsCorrcoef(0.5 if args.n_classes == 1 else None),
            'mc4': MatthewsCorrcoef(0.4 if args.n_classes == 1 else None),
            'mc45': MatthewsCorrcoef(0.45 if args.n_classes == 1 else None),
            'mc55': MatthewsCorrcoef(0.55 if args.n_classes == 1 else None),
            'mc6': MatthewsCorrcoef(0.6 if args.n_classes == 1 else None),
            'acc': Accuracy(0.5 if args.n_classes == 1 else None),
        }
    else:
        metrics = {
            'score': MatthewsCorrcoef(None),
            'acc': Accuracy(None),
        }

    history.update({k: {v: [] for v in ['train', 'dev']} for k in metrics})

    base_name = f'{args.encoder}_b{args.batch_size}_{args.opt}_lr{args.lr}_f{args.fold}_fps{args.fps}_s{args.size}'
    work_dir = Path(args.work_dir) / base_name
    if args.local_rank == 0 and not work_dir.exists():
        work_dir.mkdir(parents=True)

    # Optionally load model from a checkpoint
    if args.load:
        def _load():
            path_to_load = Path(args.load)
            if path_to_load.is_file():
                print(f"=> loading model '{path_to_load}'")
                checkpoint = torch.load(path_to_load, map_location=lambda storage, loc: storage.cuda(args.gpu))
                model.load_state_dict(checkpoint['state_dict'])
                if args.fp16 and checkpoint['amp'] is not None:
                    apex.amp.load_state_dict(checkpoint['amp'])
                print(f"=> loaded model '{path_to_load}'")
            else:
                print(f"=> no model found at '{path_to_load}'")

        _load()
        
        
    scheduler = None
    if args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.T_max, eta_min=max(args.lr * 1e-2, 1e-6))

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def _resume():
            nonlocal history, best_score
            path_to_resume = Path(args.resume)
            if path_to_resume.is_file():
                print(f"=> loading resume checkpoint '{path_to_resume}'")
                checkpoint = torch.load(path_to_resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
                args.start_epoch = checkpoint['epoch'] + 1
                history = checkpoint['history']
                best_score = checkpoint['best_score']
                model.load_state_dict(checkpoint['state_dict'])
                opt.load_state_dict(checkpoint['opt_state_dict'])
                args.ft = checkpoint['args'].ft
                if 'sched_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['sched_state_dict'])
                if args.fp16 and checkpoint['amp'] is not None:
                    apex.amp.load_state_dict(checkpoint['amp'])
                print(f"=> resume from checkpoint '{path_to_resume}' (epoch {checkpoint['epoch']})")
            else:
                print(f"=> no checkpoint found at '{args.resume}'")

        _resume()
    history.update({k: {v: [] for v in ['train', 'dev']} for k in metrics if k not in history})

    path_to_data = Path(args.data)
    train_gps, dev_gps = get_data_groups(path_to_data / args.csv, args)

    if args.pretrain is not None:
        train_ds = CloudsDSPretrain(
            train_gps,
            root=path_to_data / args.lmdb,
            pretrain_root=path_to_data / args.pretrain,
            transform=train_transform,
            fps=args.fps,
            size=args.size,
            pretrain_root2=path_to_data / args.pretrain2 if args.pretrain2 is not None else None,
        )
    else:
        train_ds = CloudsDS(train_gps, root=path_to_data / args.lmdb, transform=train_transform, fps=args.fps, size=args.size)
    dev_ds = CloudsDS(dev_gps, root=path_to_data / args.lmdb, transform=dev_transform, fps=args.fps, size=args.size)

    train_sampler = None
    dev_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
        dev_sampler = torch.utils.data.distributed.DistributedSampler(dev_ds)

    batch_size = args.batch_size
    num_workers = min(batch_size, args.workers)
    train_loader = torch.utils.data.DataLoader(train_ds,
                                               batch_size=batch_size,
                                               shuffle=train_sampler is None,
                                               sampler=train_sampler,
                                               num_workers=num_workers,
                                               collate_fn=collate_fn_3d if args.model == 'cnn3d' else collate_fn,
                                               pin_memory=True)

    dev_loader = torch.utils.data.DataLoader(dev_ds,
                                             batch_size=batch_size,  # 20, 27
                                             shuffle=False,
                                             sampler=dev_sampler,
                                             num_workers=num_workers,
                                             collate_fn=collate_fn_3d if args.model == 'cnn3d' else collate_fn,
                                             pin_memory=True)

    saver = lambda path: torch.save({
        'epoch': epoch,
        'best_score': best_score,
        'history': history,
        'state_dict': model.state_dict(),
        'opt_state_dict': opt.state_dict(),
        'sched_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'amp': apex.amp.state_dict() if args.fp16 else None,
        'args': args,
    }, path)

    teachers = None
    if args.teachers is not None:
        teachers = [
            torch.jit.load(str(p)).cuda().eval()
            for p in Path(args.teachers).rglob('*.pt')
        ]

        if args.distributed:
            for i in range(len(teachers)):
                teachers[i] = apex.parallel.DistributedDataParallel(teachers[i], delay_allreduce=True)

        print(f'#teachers: {len(teachers)}')

    for epoch in range(args.start_epoch, args.epochs + 1):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        for metric in metrics.values():
            metric.clean()
        loss = epoch_step(train_loader, f'[ Training {epoch}/{args.epochs}.. ]',
                          model=model, criterion=criterion, metrics=metrics, opt=opt,
                          batch_accum=args.batch_accum, teachers=teachers)
        history['loss']['train'].append(loss)
        for k, metric in metrics.items():
            history[k]['train'].append(metric.evaluate())

        if not args.ft:
            with torch.no_grad():
                for metric in metrics.values():
                    metric.clean()
                loss = epoch_step(dev_loader, f'[ Validating {epoch}/{args.epochs}.. ]',
                                  model=model, criterion=criterion, metrics=metrics, opt=None)
                history['loss']['dev'].append(loss)
                for k, metric in metrics.items():
                    preds = all_gather(metric.preds)
                    tgts = all_gather(metric.tgts)
                    metric.preds = list(it.chain(*preds))
                    metric.tgts = list(it.chain(*tgts))

                    history[k]['dev'].append(metric.evaluate())
        else:
            history['loss']['dev'].append(loss)
            for k, metric in metrics.items():
                history[k]['dev'].append(metric.evaluate())

        if scheduler is not None:
            scheduler.step()

        if args.local_rank == 0:
            saver(work_dir / 'last.pth')
            if epoch in [515, 705, 798, 915, 1115]:
                saver(work_dir / f'chkp_{epoch}.pth')

            if history['score']['dev'][-1] > best_score:
                best_score = history['score']['dev'][-1]
                saver(work_dir / 'best.pth')

            plot_hist(history, work_dir)


if __name__ == '__main__':
    main()
