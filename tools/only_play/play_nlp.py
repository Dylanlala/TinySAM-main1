import torch
import torch.nn as nn

# 输入张量
x=torch.randn(4,10,64)
layer_norm=nn.LayerNorm(normalized_shape=64)
y=layer_norm(x)
print(y.shape)

class MyLayerNorm(nn.Module):
    def __init__(self, features, eps=1e-5):
        super().__init__()
        self.gamma=nn.Paramter(torch.ones(features))
        self.beta=nn.Parameter(torch.zeros(features))
        self.eps=eps
    
    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        var=x.var(-1,keepdim=True,unbiased=False)
        x_norm=(x-mean)/torch.sqrt(var+self.eps)
        return self.gamma*x_norm+self.beta

layer=nn.TransformerEncoderLayer(d_model=512,nhead=8,norm_first=True)
