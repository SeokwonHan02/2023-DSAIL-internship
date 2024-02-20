import pandas as pd
import argparse
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from model import MF
from model import MF_bias
from load_data import load_dataset
from load_data import split_load
from train import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./ml-100k/')
    parser.add_argument('--dim', type=int, default=100)
    parser.add_argument('--learn_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--bias', type=bool, default=True)
    parser.add_argument('--weight_decay', type=float, default=0.00005)

    args = parser.parse_args()

    df = load_dataset(args.data_path)
    all_user = set(df['user_id'].unique())
    all_item = set(df['item_id'].unique())

    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)

    train_dataset, train_dataloader = split_load(args, train_df)
    test_dataset, test_dataloader = split_load(args, test_df)

    train_ratings = torch.tensor(train_df['rating'].values, dtype=torch.float32)
    mean_rating = train_ratings.mean().item()

    if args.bias:
        model = MF_bias(len(all_user), len(all_item), args.dim)
    else :
        model = MF(len(all_user), len(all_item), args.dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay = args.weight_decay)

    train(args, train_dataloader, train_dataset, test_dataloader, test_dataset, model, criterion, optimizer, device)
