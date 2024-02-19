import torch 
import torch.nn as nn

class Factorization_Machine(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Factorization_Machine, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        self.V = nn.Parameter(torch.randn(input_dim, output_dim))
        
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.normal_(self.V, std=0.01)
    
    def forward(self, x):
        output_1 = torch.pow(torch.matmul(x, self.V),2)
        output_2 = torch.matmul(torch.pow(x,2), torch.pow(self.V,2))

        x = self.linear(x).squeeze() + 0.5 * torch.sum(output_1 - output_2, dim=1)

        return x
