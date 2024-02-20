import torch
import torch.nn as nn

class MF(nn.Module):
  def __init__(self, user_num, item_num, dim):
    super(MF, self).__init__()
    self.user_num = user_num
    self.item_num = item_num
    self.dim = dim
    
    self.user_embed = nn.Embedding(user_num, dim)
    self.item_embed = nn.Embedding(item_num, dim)

  def forward(self, user_id, item_id):
    user_embed = self.user_embed(user_id - 1)
    item_embed = self.item_embed(item_id - 1)

    prediction = torch.sum(user_embed * item_embed, dim=1)

    return prediction

class MF_bias(nn.Module):
  def __init__(self, user_num, item_num, dim):
    super(MF_bias, self).__init__()
    self.user_num = user_num
    self.item_num = item_num
    self.dim = dim
        
    # Call the parent class constructor
    super(MF_bias, self).__init__()
        
    self.user_embedding = nn.Embedding(user_num, dim)
    self.item_embedding = nn.Embedding(item_num, dim)
    self.user_bias = nn.Embedding(user_num, 1)
    self.item_bias = nn.Embedding(item_num, 1)
        
    # Initialize biases
    nn.init.zeros_(self.user_bias.weight)
    nn.init.zeros_(self.item_bias.weight)

  def forward(self, user_ids, item_ids):
    user_embed = self.user_embedding(user_ids - 1)
    item_embed = self.item_embedding(item_ids - 1)
    user_bias = self.user_bias(user_ids - 1).squeeze()
    item_bias = self.item_bias(item_ids - 1).squeeze()

    # Calculate prediction
    prediction = torch.sum(user_embed * item_embed, dim=1) + user_bias + item_bias

    return prediction