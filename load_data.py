import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_dataset(dpath):
    df = pd.read_csv(os.path.join(dpath,'u.data'), sep='\t', header=None)
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df.drop(['timestamp'], axis=1, inplace=True)

    df['user_id'] = df['user_id'].astype('int')
    df['item_id'] = df['item_id'].astype('int')
    df['rating'] = df['rating'].astype('int')

    return df

def split_load(args, df):

    user_ids = torch.tensor(df['user_id'].values)
    item_ids = torch.tensor(df['item_id'].values)
    ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    dataset = TensorDataset(user_ids, item_ids, ratings)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataset, dataloader