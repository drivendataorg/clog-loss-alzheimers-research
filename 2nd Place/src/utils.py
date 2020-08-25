import pandas as pd
from sklearn.model_selection import StratifiedKFold


def split_df(df, args):
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)

    df['fold'] = -1

    fold_idx = len(df.columns) - 1
    
    y = (df.project_id + df.stalled.apply(str))

    for i, (_, dev_index) in enumerate(skf.split(range(len(df)), y.values)):
        df.iloc[dev_index, fold_idx] = i
        
    return df[df.fold != args.fold].reset_index(drop=True), df[df.fold == args.fold].reset_index(drop=True)


def get_data_groups(path, args):
    df = pd.read_csv(path)
    df_labels = pd.read_csv(path.parent / 'train_labels.csv')
    df = pd.merge(df, df_labels, on='filename')
    
    train, dev = split_df(df[df.micro].copy(), args)
    
    if args.ft:
        train = df[df.micro].copy()
    
    if args.pretrain is not None:
        train = pd.concat([train, df[df.tier1 & ~df.micro].copy()])
        
        assert len(train) == len(set(train.filename))
        
    if args.pretrain2 is not None:
        tier2 = df[~df.tier1 & (df.stalled == 1) & (df.crowd_score > 0.6)].copy()
        train = pd.concat([train, tier2])

    return train, dev
