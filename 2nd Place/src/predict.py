import argparse
from pathlib import Path

import apex
import numpy as np
import torch
import tqdm

from model import get_model
from dataset import CloudsDS, dev_transform, collate_fn, collate_fn_3d
from metric import Accuracy, MatthewsCorrcoef
from utils import get_data_groups


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data')
    parser.add_argument('--load', type=str, required=True,
                        help='Load model')
    parser.add_argument('--save', type=str, default='',
                        help='Save predictions')
    parser.add_argument('--tta', type=int, default=0,
                        help='Test time augmentations')

    return parser.parse_args()
    

def epoch_step(loader, desc, model, metrics):
    model.eval()

    pbar = tqdm.tqdm(total=len(loader), desc=desc, leave=False, mininterval=2)
    loc_targets, loc_preds = [], []
    
    for x, y in loader:
        x = [_x.cuda(args.gpu) for _x in x] if isinstance(x, list) else x.cuda(args.gpu)
        y = y.cuda(args.gpu)
        
        masks = []
        logits = model(x)
        if args.n_classes == 1:
            masks.append(torch.sigmoid(logits).cpu().numpy())
        else:
            masks.append(torch.softmax(logits, dim=-1).cpu().numpy())
        
        if args.tta > 0:
            logits = model(torch.flip(x, dims=[-1]))
            if args.n_classes == 1:
                masks.append(torch.sigmoid(logits).cpu().numpy())
            else:
                masks.append(torch.softmax(logits, dim=-1).cpu().numpy())
            
        if args.tta > 1:
            logits = model(torch.flip(x, dims=[-2]))
            if args.n_classes == 1:
                masks.append(torch.sigmoid(logits).cpu().numpy())
            else:
                masks.append(torch.softmax(logits, dim=-1).cpu().numpy())
            
        if args.tta > 2:
            logits = model(torch.flip(x, dims=[-1, -2]))
            if args.n_classes == 1:
                masks.append(torch.sigmoid(logits).cpu().numpy())
            else:
                masks.append(torch.softmax(logits, dim=-1).cpu().numpy())

        trg = y.cpu().numpy()
        loc_targets.extend(trg)
        preds = np.mean(masks, 0)
        loc_preds.extend(preds)
    
        for metric in metrics.values():
            metric.update(preds, trg)
        
        torch.cuda.synchronize()

        if args.local_rank == 0:
            pbar.set_postfix(**{
                k: f'{metric.evaluate():.4}' for k, metric in metrics.items()
            })
            pbar.update()

    pbar.close()
    
    return loc_targets, loc_preds

    
def main():
    global args
    
    args = parse_args()
    print(args)

    torch.backends.cudnn.benchmark = True

    args.gpu = 0
    assert torch.backends.cudnn.enabled, 'Amp requires cudnn backend to be enabled.'
    
    to_save = args.save
    path_to_load = Path(args.load)
    if path_to_load.is_file():
        print(f"=> Loading checkpoint '{path_to_load}'")
        checkpoint = torch.load(path_to_load, map_location=lambda storage, loc: storage.cuda(args.gpu))
        print(f"=> Loaded checkpoint '{path_to_load}'")
    else:
        raise

    tta = args.tta
    args = checkpoint['args']
    args.tta = tta
    print(args)

    model = get_model(args.model, args.encoder, args.n_classes)
    
    model.cuda()
     
    # Initialize Amp. Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.

    assert args.fp16 == False, "torch script doesn't work with amp"
    if args.fp16:
        model = apex.amp.initialize(model,
                                    opt_level=args.opt_level,
                                    keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                    loss_scale=args.loss_scale
                                   )
    
    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
#     if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with 
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
#         model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        
    work_dir = path_to_load.parent
    
    import copy
    state_dict = copy.deepcopy(checkpoint['state_dict'])
    for p in checkpoint['state_dict']:
        if p.startswith('module.'):
            state_dict[p[len('module.'):]] = state_dict.pop(p)

    model.load_state_dict(state_dict)

    if args.fp16 and checkpoint['amp'] is not None:
        print('amp state dict')
        apex.amp.load_state_dict(checkpoint['amp'])
    
    x = torch.rand(1, 1, 64, 160, 160).cuda()
    model = model.eval()
    if 'efficientnet' in args.encoder:
        model.features.set_swish(memory_efficient=False)
    with torch.no_grad():
        traced_model = torch.jit.trace(model, x)
    
    traced_model.save(str(work_dir / f'model_{path_to_load.stem}.pt'))
    del traced_model
    del model
    
    path_to_data = Path(args.data)
    train_gps, dev_gps = get_data_groups(path_to_data / args.csv, args)

    dev_ds = CloudsDS(dev_gps, root=path_to_data / args.lmdb, transform=dev_transform, fps=args.fps, size=args.size)

    dev_loader = torch.utils.data.DataLoader(dev_ds,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             sampler=None,
                                             num_workers=args.batch_size,
                                             collate_fn=collate_fn if args.model == 'lstm' else collate_fn_3d,
                                             pin_memory=True)

    if args.n_classes == 1:
        metrics = {
            'score': MatthewsCorrcoef(0.5 if args.n_classes == 1 else None),
            'mc4': MatthewsCorrcoef(0.4 if args.n_classes == 1 else None),
            'mc45': MatthewsCorrcoef(0.45 if args.n_classes == 1 else None),
            'mc55': MatthewsCorrcoef(0.55 if args.n_classes == 1 else None),
            'mc6': MatthewsCorrcoef(0.6 if args.n_classes == 1 else None),
            'mc65': MatthewsCorrcoef(0.65 if args.n_classes == 1 else None),
            'mc7': MatthewsCorrcoef(0.7 if args.n_classes == 1 else None),
            'acc': Accuracy(0.5 if args.n_classes == 1 else None),
        }
    else:
        metrics = {
            'score': MatthewsCorrcoef(None),
            'acc': Accuracy(None),
        }
    
    model = torch.jit.load(str(work_dir / f'model_{path_to_load.stem}.pt')).cuda().eval()

    with torch.no_grad():
        for metric in metrics.values():
            metric.clean()

        trgs, preds = epoch_step(dev_loader, f'[ Validating dev.. ]',
                                 model=model,
                                 metrics=metrics)
        for key, metric in metrics.items():
            print(f'{key} dev {metric.evaluate()}')

    if str(to_save) == '':
        return

    to_save = Path(to_save)
    if not to_save.exists():
        to_save.mkdir(parents=True)

    dev_gps['preds'] = np.array(preds).ravel()
    dev_gps.to_csv(to_save / f'preds_f{args.fold}.csv', index=False)


if __name__ == '__main__':
    main()
