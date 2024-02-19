import os 
import numpy as np 
import pandas as pd 
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset , DataLoader

def age_map(x):
    x = int(x)
    if x < 20:
        return '10'
    elif x >= 20 and x < 30:
        return '20'
    elif x >= 30 and x < 40:
        return '30'
    elif x >= 40 and x < 50:
        return '40'
    elif x >= 50 and x < 60:
        return '50'
    else:
        return '60'

def load_dataset(dpath) :
    df = pd.read_csv(os.path.join(dpath,'u.data'), sep='\t', header=None)
    df.columns = ['user_id', 'item_id', 'rating', 'timestamp']

    # For each unique user ID 'j', assign index 'i'
    user2idx = {j:i for i,j in enumerate(df.user_id.unique())}
    # For each unique item ID 'j', assign index 'i'
    item2idx = {j:i for i,j in enumerate(df.item_id.unique())}

    df['user_id'] = df['user_id'].map(user2idx)
    df['item_id'] = df['item_id'].map(item2idx)

    movies_df = pd.read_csv(os.path.join(dpath,'u.item'), sep='|', header=None, encoding='latin-1')
    movies_df.columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date',
                        'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 
                        'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                        'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                        'Thriller', 'War', 'Western']

    users_df = pd.read_csv(os.path.join(dpath,'u.user'), sep='|', encoding='latin-1', header=None)
    users_df.columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']

    users_df['age'] = users_df['age'].apply(age_map)

    movies_df.drop(['movie_title', 'release_date', 'video_release_date', 'IMDb_URL'], axis=1, inplace=True)
    movies_df['movie_id'] = movies_df['movie_id'].map(item2idx)
    users_df['user_id'] = users_df['user_id'].map(user2idx)

    df.rename(columns={'item_id':'movie_id'}, inplace=True)

    df = pd.merge(df, movies_df,how='left', on = 'movie_id')
    df = pd.merge(df, users_df, how='left',on = 'user_id')
    
    df.drop(['timestamp', 'zip_code'], axis=1, inplace=True)
    df['user_id'] = df['user_id'].astype('category')
    df['movie_id'] = df['movie_id'].astype('category')
    df['age'] = df['age'].astype('category')
    df['gender'] = df['gender'].astype('category')
    df['occupation'] = df['occupation'].astype('category')
    
    # Drop the 'rating' column
    df_without_rating = df.drop('rating', axis=1)

    # Perform one-hot encoding on the DataFrame without the 'rating' column
    fm_df_without_rating = pd.get_dummies(df_without_rating)

    # Concatenate the 'rating' column back to the resulting DataFrame
    fm_df = pd.concat([fm_df_without_rating, df['rating']], axis=1)
    
    return fm_df

def load_data_split(args, df) :
    train_X, test_X, train_y, test_y = train_test_split(df.loc[:, df.columns != 'rating'], df['rating'], test_size=args.test_size, random_state=args.rand_seed)

    train_X = train_X.astype(np.float64)
    train_y = train_y.astype(np.float64)
    test_X = test_X.astype(np.float64)
    test_y = test_y.astype(np.float64)

    train_dataset = TensorDataset(torch.Tensor(np.array(train_X)), torch.Tensor(np.array(train_y)))
    test_dataset = TensorDataset(torch.Tensor(np.array(test_X)), torch.Tensor(np.array(test_y)))

    # Read the data set by dividing it into mini batches of the desired size
    train_dataload = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataload = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    return train_dataload, test_dataload