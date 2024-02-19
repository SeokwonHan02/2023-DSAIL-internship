import argparse
import torch
import torch.nn as nn

from load_data import load_data_split
from load_data import load_dataset
from model import Factorization_Machine
from train import train

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='./ml-100k/')
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--epoch_num', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learn_rate', type=float, default=0.001)
    parser.add_argument('--factor_dim', type=int, default=10)

    args = parser.parse_args()
    
    df = load_dataset(args.data_path)
    train_dataload, test_dataload = load_data_split(args, df)

    input_dim = len(df.columns) - 1
    model = Factorization_Machine(input_dim, args.factor_dim)
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    
    train(args, model, train_dataload, criterion, optimizer, device)