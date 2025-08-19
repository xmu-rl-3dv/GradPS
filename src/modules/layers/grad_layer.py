import torch as th
import torch.nn as nn


class GradLayer(nn.Module):
    def __init__(self, dim, name, agent=1):
        super(GradLayer, self).__init__()
        self.dim = dim
        self.agent = agent
        self.weight = nn.Parameter(th.ones(agent, dim))
        # self.weight.requires_grad = False
        self.name = name

    def forward(self, x):
        # print('pre',x)
        # avg = x.mean()
        # x[x<0.01*avg] = 0
        # print(x.shape,self.weight.data.shape)
        # if len(self.weight.shape) >= 2:
        #     return torch.mul(x, self.weight.repeat(x.shape[0] // self.weight.shape[0], 1))
        w = x.view(-1, self.agent, self.dim)
        q = w * self.weight

        z = q.view(-1, self.dim)
        return z
        # return x * self.weight