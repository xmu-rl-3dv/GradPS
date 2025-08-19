import torch.nn as nn
import torch.nn.functional as F
from modules.layers.grad_layer import GradLayer
from modules.layers.share_layer import ShareLayer


class RNNAgentShare(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgentShare, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.net = nn.Sequential(ShareLayer(input_shape, args.rnn_hidden_dim, 'C1', agent=args.n_agents, K=min(args.n_agents, args.cluster_k)),
                                 nn.ReLU(inplace=True),
                                 GradLayer(args.rnn_hidden_dim, 'C1', agent=args.n_agents),)

        # self.net = nn.Sequential(nn.Linear(input_shape, args.rnn_hidden_dim),
        #                          nn.ReLU(inplace=True),
        #                          GradLayer(args.rnn_hidden_dim, 'C1', agent=args.n_agents),
        #                          nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim),
        #                          nn.ReLU(inplace=True),
        #                          GradLayer(args.rnn_hidden_dim, 'C2', agent=args.n_agents),)

        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        if len(inputs.shape)==2:
            b = 1
            a, e = inputs.size()
        else:
            b, a, e = inputs.size()
        x = self.net(inputs.view(-1, e))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q.view(b, a, -1), h.view(b, a, -1)