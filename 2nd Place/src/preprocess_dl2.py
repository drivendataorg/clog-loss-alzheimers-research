import argparse
from pathlib import Path
import functools as f
import multiprocessing

from awscli.customizations.s3.utils import split_s3_bucket_key
import boto3
import cv2
import pickle
import numpy as np
import pandas as pd
import lmdb
import tqdm


def find_bucket_key(s3_path):
    """
    This is a helper function that given an s3 path such that the path is of
    the form: bucket/key
    It will return the bucket and the key represented by the s3 path
    """
    s3_components = s3_path.split('/')
    bucket = s3_components[0]
    s3_key = ""
    if len(s3_components) > 1:
        s3_key = '/'.join(s3_components[1:])

    return bucket, s3_key


def split_s3_bucket_key(s3_path):
    """Split s3 path into bucket and key prefix.
    This will also handle the s3:// prefix.
    :return: Tuple of ('bucketname', 'keyname')
    """
    if s3_path.startswith('s3://'):
        s3_path = s3_path[5:]

    return find_bucket_key(s3_path)


# https://community.drivendata.org/t/python-code-to-find-the-roi/4499
def get_rect(row, root, mode):
    _, item = row
    frames = []
    cap = cv2.VideoCapture(str(root / mode / item.filename))
    xs, ys, ws, hs = [], [], [], []
    v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(v_len): 
        success, image = cap.read()
            
        mask = cv2.inRange(image, (9, 13, 104), (98, 143, 255))
        points = np.where(mask > 0)
        p2 = [p for p in zip(points[0], points[1])]
        x, y, w, h = cv2.boundingRect(np.float32(p2))
        xs.append(x)
        ys.append(y)
        ws.append(w)
        hs.append(h)
        frames.append(image[..., -1])

    cap.release()
    
    frames = np.array(frames)
    x = min(xs)
    y = min(ys)
    w = max(ws)
    h = max(hs)

    frames = frames[:, x:x + w, y:y + h]
    out = {'frames': np.array(frames).transpose(1, 2, 0)}
    
    if 'stalled' in item:
        out.update({'label': item.stalled})
    
    return item.filename.split('.')[0], out


def dl_and_get_rect(row, root, mode):
    if 'tier' in mode:
        _, item = row

        s3 = boto3.client('s3')

        url = item.url
        bucket_name, key_name = split_s3_bucket_key(url)
        fname = key_name.split('/')[-1]
        s3.download_file(bucket_name, key_name, fname)

        x, y = get_rect(row, Path(''), '')

        Path(fname).unlink()

        return x, y
    
    return get_rect(row, root, mode)


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', choices=['test', 'micro', 'tier1', 'tier2'])
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--save', type=str, required=True)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    data_root = Path(args.data)
    args.save = Path(args.save)
    mode = args.mode
    
    if mode == 'test':
        meta = pd.read_csv(data_root / 'test_metadata.csv')
    else:
        meta = pd.read_csv(data_root / 'train_metadata.csv')
        if mode == 'tier1':
            meta = meta[meta[mode] & ~meta.micro].copy()
        else:
            meta = meta[~meta.tier1].copy()
        meta = pd.merge(
            meta,
            pd.read_csv(data_root / 'train_labels.csv'),
            on='filename',
        )
        if mode == 'tier2':
            meta = meta[meta.stalled == 1].copy()
        
    with tqdm.tqdm(meta.iterrows(), total=len(meta)) as pbar:
        with multiprocessing.Pool(20) as p:
            features = dict(p.imap_unordered(func=f.partial(dl_and_get_rect, root=data_root, mode=mode), iterable=pbar))

    map_size = sum(features[f]['frames'].nbytes for f in features)
    map_size = int(1.05*map_size)
    
    env = lmdb.open(
        str(args.save / f'{mode}_croped_tier2.lmdb'),
        map_size=map_size,
    )

    with env.begin(write=True) as txn:
        txn.put(('keys').encode(), pickle.dumps(list(features.keys())))

        with tqdm.tqdm(features) as pbar:
            for key in pbar:
                txn.put(key.encode(), pickle.dumps(features[key]))
            
            
if __name__ == '__main__':
    main()
