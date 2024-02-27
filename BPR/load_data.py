import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

def load_dataset(dpath):
    df = pd.read_csv(os.path.join(dpath,'u.data'), sep='\t', header=None)
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
    df.drop(['timestamp'], axis=1, inplace=True)

    return df

def triplet(selected_df, selected_users, selected_items):
    D_S = []

    for u in selected_users:
        rated_items = selected_df[selected_df['user_id'] == u]['item_id'].values
        unrated_items = list(set(selected_items) - set(rated_items))

        for i in rated_items:
            for j in unrated_items:
                D_S.append((u, i, j))
    
    return D_S

def dataload(D_S, args):
    user = torch.tensor([t[0] for t in D_S])
    rated_items = torch.tensor([t[1] for t in D_S])
    unrated_items = torch.tensor([t[2] for t in D_S], dtype=torch.float32)

    dataset = TensorDataset(user, rated_items, unrated_items)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    return dataset, dataloader
