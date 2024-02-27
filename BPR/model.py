import torch
import torch.nn as nn

class MF_BPR(nn.Module):
    def __init__(self,user_num,item_num,dim):
        super(MF_BPR, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.embed_dim = dim

        self.user_embedding = nn.Embedding(self.user_num,self.embed_dim)
        self.item_embedding = nn.Embedding(self.item_num,self.embed_dim)
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_id, item_id):

        user_embed = self.user_embedding(user_id - 1)
        item_embed = self.item_embedding(item_id - 1)
        user_bias = self.user_bias(user_id - 1).squeeze()
        item_bias = self.item_bias(item_id - 1).squeeze()

        prediction = torch.sum(user_embed * item_embed, dim=1) + user_bias + item_bias

        return prediction

    def bpr_loss(self, user_id, rated_id, unrated_id):

        user_embed = self.user_embedding(user_id - 1)
        rated_embed = self.item_embedding(rated_id - 1)
        unrated_embed = self.item_embedding(unrated_id - 1)

        user_bias = self.user_bias(user_id - 1).squeeze()
        rated_bias = self.item_bias(rated_id - 1).squeeze()
        unrated_bias = self.item_bias(unrated_id - 1).squeeze()

        pos_scores = torch.sum(user_embed * rated_embed, dim=1) + user_bias + rated_bias
        neg_scores = torch.sum(user_embed * unrated_embed, dim=1) + user_bias + unrated_bias
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))

        return loss
