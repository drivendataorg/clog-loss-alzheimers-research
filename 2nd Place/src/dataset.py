import random

import albumentations as A
import cv2
import pickle
import torch
import lmdb


p = 0.5
albu_train = A.Compose([
    A.HorizontalFlip(p=p),
    A.VerticalFlip(p=p),
    A.RandomRotate90(p=p),

    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.RandomGamma(p=1),
    ], p=p),
    
    A.OneOf([
        A.GaussianBlur(p=1),
        A.GaussNoise(p=1),
        A.IAAAdditiveGaussianNoise(p=1),

        A.Blur(p=1),
        A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=1),
    ], p=p),

    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=150 * 0.05, alpha_affine=120 * 0.03, p=1),
        A.GridDistortion(p=1),
        A.OpticalDistortion(distort_limit=1, shift_limit=120 * 0.03, p=1),
    ], p=p),
    
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=p)
])


def to_tensor(img):
    img = img.astype('float32') / 255

    return torch.from_numpy(img).permute(2, 0, 1)


def train_transform(img):
    img = albu_train(image=img)['image']
    
    if random.random() < 0.5:
        img = img[..., ::-1]

    return to_tensor(img)


def dev_transform(img):
    return to_tensor(img)


class CloudsDS(torch.utils.data.Dataset):
    def __init__(self, df, root, transform, fps=2, size=160):
        self.df = df
        self.transform = transform
        self.fps = fps

        self.env = lmdb.open(
            str(root),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        rsz = A.Resize(size, size)
        with self.env.begin(write=False, buffers=True) as txn:
            self.features = []
            for _, item in self.df.iterrows():
                fname = item.filename.split('.')[0]
                feature = pickle.loads(txn.get(fname.encode()))
                
                frames = feature['frames']
                frames = frames[..., 1::self.fps]
                frames = rsz(image=frames)['image']

                self.features.append(
                    (feature.get('label', fname), frames)
                )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        fname, feature = self.features[index]

        frames = self.transform(feature)

        return frames, fname
    
    
class CloudsDSPretrain(torch.utils.data.Dataset):
    def __init__(self, df, root, pretrain_root, transform, fps=2, size=160, pretrain_root2=None):
        self.transform = transform
        self.fps = fps
        self.df_pos = df[df.stalled == 1].copy()
        self.df_neg = df[df.stalled == 0].copy()

        self.env = lmdb.open(
            str(root),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
        self.pretrain = lmdb.open(
            str(pretrain_root),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        
        if pretrain_root2 is not None:
            self.pretrain2 = lmdb.open(
                str(pretrain_root2),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )

        self.rsz = A.Resize(size, size)

    def __len__(self):
        return 4*len(self.df_pos)

    def __getitem__(self, index):
        if index % 4 == 0:  # pos
            item = self.df_pos.iloc[index//4]
        else:  # neg
            item = self.df_neg.sample(n=1).iloc[0]

        fname = item.filename.split('.')[0]
        if item.micro:
            with self.env.begin(write=False, buffers=True) as txn:
                feature = pickle.loads(txn.get(fname.encode()))
        elif item.tier1:
            with self.pretrain.begin(write=False, buffers=True) as txn:
                feature = pickle.loads(txn.get(fname.encode()))
        elif hasattr(self, 'pretrain2'):
            with self.pretrain2.begin(write=False, buffers=True) as txn:
                feature = pickle.loads(txn.get(fname.encode()))

        frames = feature['frames']
        frames = frames[..., ::self.fps]
        frames = self.rsz(image=frames)['image']
        if 'label' in feature:
            fname = feature['label']
        
        frames = self.transform(frames)

        return frames, fname


class CloudsDSTest(torch.utils.data.Dataset):
    def __init__(self, df, root, transform, fps=2, size=160):
        self.df = df
        self.transform = transform
        self.fps = fps
        self.rsz = A.Resize(size, size)

        self.env = lmdb.open(
            str(root),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        fname = item.filename.split('.')[0]
        with self.env.begin(write=False, buffers=True) as txn:
            feature = pickle.loads(txn.get(fname.encode()))

        img = self.rsz(image=feature['frames'][::self.fps])['image']
        frames = self.transform(img)
        if 'label' in feature:
            fname = feature['label']

        return frames, fname


def collate_fn(x):
    x, y = list(zip(*x))

    return x, torch.tensor(y).float().unsqueeze(1)


def collate_fn_3d(x):
    x, y = list(zip(*x))

    max_len = max(map(len, x))
    x = torch.stack([
        torch.nn.functional.pad(_x, (0, 0, 0, 0, 0, max_len - len(_x)))
        for _x in x
    ])  # [b, seq, h, w]

    return x.unsqueeze(1), torch.tensor(y).float().unsqueeze(1)
