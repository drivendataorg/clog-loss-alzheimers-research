import argparse
from pathlib import Path
import functools as f
import multiprocessing

import cv2
import pickle
import numpy as np
import pandas as pd
import lmdb
import tqdm


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


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', choices=['test', 'micro'])
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
        meta = meta[meta[mode]].copy()
        meta = pd.merge(
            meta[meta[mode]].copy(),
            pd.read_csv(data_root / 'train_labels.csv'),
            on='filename',
        )
        
    with tqdm.tqdm(meta.iterrows(), total=len(meta)) as pbar:
        with multiprocessing.Pool(40) as p:
            features = dict(p.imap_unordered(func=f.partial(get_rect, root=data_root, mode=mode), iterable=pbar))

    map_size = sum(features[f]['frames'].nbytes for f in features)
    map_size = int(1.1*map_size)
    
    env = lmdb.open(
        str(args.save / f'{mode}_croped2.lmdb'),
        map_size=map_size,
    )

    with env.begin(write=True) as txn:
        txn.put(('keys').encode(), pickle.dumps(list(features.keys())))

        with tqdm.tqdm(features) as pbar:
            for key in pbar:
                txn.put(key.encode(), pickle.dumps(features[key]))
            
            
if __name__ == '__main__':
    main()
