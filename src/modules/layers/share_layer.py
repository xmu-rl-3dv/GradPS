import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class ShareLayer(nn.Module):
    def __init__(self, input_dim, output_dim, name, agent, K):
        super(ShareLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.agent = agent
        self.K = K

        self.cluster = nn.Parameter(th.zeros(output_dim, agent, K))
        self.cluster.data[:, :, 0] = 1  # default first group
        self.cluster.requires_grad = False

        self.root = nn.Linear(input_dim, output_dim)

        self.weight = nn.Parameter(th.zeros(output_dim, K, input_dim))  # (output, K, input)
        self.weight.data[:, 0, :] = self.root.weight.data
        self.bias = self.root.bias

        # self.cluster.data[1, 0:2, 0] = 1
        # self.cluster.data[1, 2:3, 1] = 1
        # self.cluster.data[1, 3:, 2] = 1
        #
        # self.cluster.data[2, 0:1, 0] = 1
        # self.cluster.data[2, 1:2, 1] = 1
        # self.cluster.data[2, 2:, 2] = 1
        # for i in range(K):
        #     self.weight.data[:,i,:] *= (i+1)

    def open_cluster(self, dim_list, group):
        for dim in dim_list:
            # print(self.cluster)
            self.weight.data[dim, :, :] = self.weight.data[dim, 0, :]
            self.cluster.data = F.one_hot(group).float()
            # print(self.cluster)

    def close_cluster(self, dim_list, value):
        for dim in dim_list:
            # add = th.sum(value[:, dim])
            # if add == 0:
            #     weighting = 1
            # else:
            #     weighting = torch.matmul(value[:, dim] / add, self.cluster[dim]).unsqueeze(1)  # (K,1)

            avg = th.mean(self.weight.data[dim], dim=0)

            self.weight.data[dim] = 0
            self.weight.data[dim, 0] = avg

            self.cluster.data[dim, :, :] = 0
            self.cluster.data[dim, :, 0] = 1

    def forward(self, x):
        w = x.view(-1, self.agent, 1, self.input_dim)  # (bs, agent, 1, input)

        r = th.matmul(self.cluster, self.weight).permute(1, 2, 0)  # （agent,input,output)

        q = th.matmul(w, r) + self.bias

        z = q.view(-1, self.output_dim)

        return z
