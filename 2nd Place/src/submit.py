import argparse
from pathlib import Path

import torch
import tqdm
import pandas as pd

from dataset import CloudsDSTest, dev_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to data')
    parser.add_argument('--lmdb', type=str, required=True,
                        help='Path to data')
    parser.add_argument('--exp', type=str, required=True,
                        help='Path to models checkpoints in jit format')
    parser.add_argument('--submit-path', type=str, required=True)

    parser.add_argument('--n-parts', type=int, default=1)
    parser.add_argument('--part', type=int, default=0)
    parser.add_argument('--tta', type=int, default=1)
    parser.add_argument('--thresh', type=float, default=0.5)

    parser.add_argument('--batch-size', type=int, default=1)

    return parser.parse_args()


def collate3d(x):
    x, y = list(zip(*x))

    max_size = max(map(len, x))
    x = torch.stack([
        torch.nn.functional.pad(_x, (0, 0, 0, 0, 0, max_size - len(_x)))
        for _x in x
    ])  # [b, seq, h, w]

    return x.unsqueeze(1), y


def flip(img, dims):
    return [torch.flip(_img, dims=dims) for _img in img] if isinstance(img, list) else torch.flip(img, dims=dims)

def rot90(img):
    return [torch.rot90(_img, 1, [-2, -1]) for _img in img] if isinstance(img, list) else torch.rot90(img, 1, [-2, -1])


def main():
    args = parse_args()
    print(args)
    data_root = Path(args.data)

    test_anns = pd.read_csv(data_root / args.csv)

    n = len(test_anns)
    k = n//args.n_parts
    start = args.part*k
    end = k*(args.part + 1) if args.part + 1 != args.n_parts else n
    test_anns = test_anns.iloc[start:end].copy()
    print(f'test size: {len(test_anns)}')

    models = [
        torch.jit.load(str(p)).cuda().eval()
        for p in Path(args.exp).rglob('*.pt')
    ]
#     models = []
#     for path_to_load in Path(args.exp).rglob('best.pth'):
#         if path_to_load.is_file():
#             print(f"=> Loading checkpoint '{path_to_load}'")
#             checkpoint = torch.load(path_to_load, map_location=lambda storage, loc: storage.cuda())
#             print(f"=> Loaded checkpoint '{path_to_load}'")
#         else:
#             raise

#         args_chkp = checkpoint['args']

#         state_dict = copy.deepcopy(checkpoint['state_dict'])
#         for p in checkpoint['state_dict']:
#             if p.startswith('module.'):
#                 state_dict[p[len('module.'):]] = state_dict.pop(p)

#         if not hasattr(args_chkp, 'model'):
#             args_chkp.model = 'lstm'
#         if not hasattr(args_chkp, 'fps'):
#             args_chkp.fps = 2
#         if not hasattr(args_chkp, 'size'):
#             args_chkp.size = 128+32

#         model = get_model(args_chkp.model, args_chkp.encoder, args_chkp.n_classes).cuda().eval()
#         model.load_state_dict(state_dict)
#         models.append(model)

    batch_size = args.batch_size
    ds = CloudsDSTest(test_anns, data_root / args.lmdb, dev_transform, fps=1, size=160)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=batch_size,
        shuffle=False,
        collate_fn=collate3d,  # collate,
        pin_memory=True,
    )

    n_models = len(models)
    n_augs = n_models * args.tta
    print(f'#models: {n_models}, #augs: {n_augs}')

    def get_submit():
        masks = torch.zeros((batch_size, 1), dtype=torch.float32, device='cuda')
        submit = pd.DataFrame()
        with torch.no_grad():
            with tqdm.tqdm(loader, mininterval=2) as pbar:
                for img, anns in pbar:
                    bs = len(img)
                    img = [_img.cuda() for _img in img] if isinstance(img, list) else img.cuda()

                    masks.zero_()
                    for model in models:
                        mask = model(img)
                        masks[:bs] += torch.sigmoid(mask)

                        # vertical flip
                        if args.tta > 1:
                            mask = model(flip(img, dims=[-1]))
                            masks[:bs] += torch.sigmoid(mask)

                        # horizontal flip
                        if args.tta > 2:
                            mask = model(flip(img, dims=[-2]))
                            masks[:bs] += torch.sigmoid(mask)

                        if args.tta > 3:
                            # vertical + horizontal flip
                            mask = model(flip(img, dims=[-1, -2]))
                            masks[:bs] += torch.sigmoid(mask)

                        if args.tta > 7:
                            img_r90 = rot90(img)

                            mask = model(img_r90)
                            masks[:bs] += torch.sigmoid(mask)

                            mask = model(flip(img_r90, dims=[-1]))
                            masks[:bs] += torch.sigmoid(mask)

                            mask = model(flip(img_r90, dims=[-2]))
                            masks[:bs] += torch.sigmoid(mask)

                            mask = model(flip(img_r90, dims=[-1, -2]))
                            masks[:bs] += torch.sigmoid(mask)

                        if args.tta > 15:
                            img = flip(img, dims=[-3])
                            mask = model(img)
                            masks[:bs] += torch.sigmoid(mask)

                            # vertical flip
                            mask = model(flip(img, dims=[-1]))
                            masks[:bs] += torch.sigmoid(mask)

                            # horizontal flip
                            mask = model(flip(img, dims=[-2]))
                            masks[:bs] += torch.sigmoid(mask)

                            # vertical + horizontal flip
                            mask = model(flip(img, dims=[-1, -2]))
                            masks[:bs] += torch.sigmoid(mask)

                            img_r90 = rot90(img)

                            mask = model(img_r90)
                            masks[:bs] += torch.sigmoid(mask)

                            mask = model(flip(img_r90, dims=[-1]))
                            masks[:bs] += torch.sigmoid(mask)

                            mask = model(flip(img_r90, dims=[-2]))
                            masks[:bs] += torch.sigmoid(mask)

                            mask = model(flip(img_r90, dims=[-1, -2]))
                            masks[:bs] += torch.sigmoid(mask)

                    masks /= n_augs
                    for mask, annotation in zip(masks, anns):
                        mask = mask.cpu().numpy().astype('float32')

                        if args.thresh is None:
                            m = mask.argmax()
                        else:
#                             m = (mask > args.thresh).astype('uint8')
                            m = mask

                        sub = pd.DataFrame({
                            'filename': f'{annotation}.mp4',
                            'stalled': m,
                        })

                        submit = submit.append(sub)

        return submit

    submit = get_submit()
    submit.to_csv(args.submit_path, index=False)


if __name__ == '__main__':
    main()
