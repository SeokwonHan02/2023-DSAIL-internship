import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


from load_data import load_dataset
from load_data import triplet
from load_data import dataload
from model import MF_BPR
from train import AUC
from train import train
from train import evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./ml-100k/')
    parser.add_argument('--dim', type=int, default=50)
    parser.add_argument('--learn_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--epoch_num', type=int, default=50)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--user_num', type=int, default=200)
    parser.add_argument('--item_num', type=int, default=400)

    args = parser.parse_args()

    df = load_dataset(args.data_path)

    selected_users = np.arange(1, args.user_num + 1)
    selected_items = np.arange(1, args.item_num + 1)

    selected_df = df[df['user_id'].isin(selected_users) & df['item_id'].isin(selected_items)].copy()
    selected_df.reset_index(drop=True, inplace=True)

    D_S = triplet(selected_df, selected_users, selected_items)
    train_D_S, test_D_S = train_test_split(D_S, test_size=args.test_size, random_state=args.rand_seed)

    train_dataset, train_dataloader = dataload(train_D_S, args)
    test_dataset, test_dataloader = dataload(test_D_S, args)

    model = MF_BPR(args.user_num, args.item_num, dim=args.dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)

    train(args, model, train_dataloader, device, optimizer)
    evaluate(model, test_dataloader, device)