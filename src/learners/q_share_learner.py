import copy

import torch
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.qatten import QattenMixer
import torch as th
import numpy
from torch.optim import RMSprop, Adam, SGD
from modules.layers.grad_layer import GradLayer
from modules.layers.share_layer import ShareLayer
from torch.nn import Linear
from torch.nn import GRU
from utils.cluster import spectral

class QShareLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.device = th.device('cuda' if args.use_cuda else 'cpu')
        self.last_target_update_episode = 0

        para = sum([numpy.prod(list(p.size())) for p in self.params])
        print(args.name, para * 4 / 1024 / 1024)

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "qatten":
                self.mixer = QattenMixer(args)
            elif args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())

            self.target_mixer = copy.deepcopy(self.mixer)

        self.grad_seq = {}
        self.neu_grad = {}
        self.neu_effi = {}
        self.value_total = {}
        self.neu_value = {}
        self.group = {}
        self.group_status = {}
        self.group_time = {}
        self.group_var = {}

        for (name, module) in self.mac.agent.named_modules():  # time need to be added
            if isinstance(module, GradLayer):  # (agent,dim)
                self.grad_seq[module.name] = th.zeros((args.share_T, module.weight.shape[0], module.weight.shape[1]),device=args.device)  # (t, agent, dim)
                self.group[module.name] = th.zeros_like(module.weight.T, device=args.device, dtype=torch.int64)
                self.group_status[module.name] = th.zeros(module.weight.shape[1], device=args.device, dtype=torch.int64)
                self.group_var[module.name] = th.zeros(module.weight.shape[1], device=args.device)
                self.group_time[module.name] = th.zeros(module.weight.shape[1], device=args.device, dtype=torch.int64)

                self.neu_value[module.name] = th.zeros_like(module.weight, device=args.device)
                self.value_total[module.name] = th.zeros_like(module.weight, device=args.device)
                self.neu_grad[module.name] = th.zeros_like(module.weight, device=args.device)
                self.neu_effi[module.name] = th.zeros((args.share_T, module.weight.shape[1]), device=args.device)

                def forward_hook(module, fea_in, fea_out):
                    fea_out = fea_out.reshape(-1, args.n_agents, fea_out.shape[1])  # t(batch*agent, dim)  avg of each time
                    self.neu_value[module.name] = self.neu_value[module.name] + th.mean(fea_out, dim=0)

                module.register_forward_hook(hook=forward_hook)

        para = sum([numpy.prod(list(p.size())) for p in self.params])
        print(args.name, para * 4 / 1024 / 1024)
        # self.share_params = [param for param in self.params if torch.all(param == 1)]

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.set_optimiser = SGD(params=self.params, lr=args.set_lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(self.mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        for key in self.neu_value.keys():
            self.neu_value[key] = th.zeros_like(self.neu_value[key], device=self.args.device)

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        # chosen_action_qvals.retain_grad()   # (batch, time, agent)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            mixer_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (mixer_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        for (name, module) in self.mac.agent.named_modules():
            if isinstance(module, GradLayer):
                self.neu_grad[module.name] += module.weight.grad.detach()
                self.value_total[module.name] += self.neu_value[module.name].detach()
                module.weight.grad = None

        self.optimiser.step()

        for (name, module) in self.mac.agent.named_modules():
            if isinstance(module, ShareLayer):
                close_dim = torch.where(self.group_status[module.name] == 2)[0]
                weight = module.weight[close_dim] # (output, cluster_k, input)
                avg = torch.mean(weight, dim=1).unsqueeze(1).detach()
                loss_set = torch.sum((weight - avg) ** 2)
                self.set_optimiser.zero_grad()
                loss_set.backward()
                self.set_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            for (name, module) in self.mac.agent.named_modules():
                if isinstance(module, ShareLayer):
                    print(self.group_status)
                    time_var = torch.mean(torch.var(module.weight.data, dim=1), dim=1) / self.group_time[module.name]
                    time_var[torch.where(self.group_status[module.name] != 1)] = float('inf')
                    self.group_var[module.name] = time_var

                    #  reset
                    reset = torch.rand(1, device=self.args.device)
                    set_dim = -1
                    if reset<self.args.reset_p*torch.count_nonzero(self.group_status[module.name]==1)/self.group_status[module.name].shape[0]:
                        set_dim = torch.argmin(self.group_var[module.name])
                        if self.group_status[module.name][set_dim] == 1:
                            self.group_status[module.name][set_dim] = 2

                    #  open
                    self.group_time[module.name] += 1
                    ready_dim = torch.where((self.group_time[module.name] > self.args.share_T)&(self.group_status[module.name] == 0))[0]
                    effi = torch.mean(self.neu_effi[module.name][:, ready_dim],dim=0)
                    # open_dim = ready_dim[torch.where(effi<0.2)[0]]
                    open_dim = ready_dim[torch.where(effi<0.5*torch.mean(effi))[0]]

                    self.group_time[module.name][open_dim] = 0
                    self.calc_conflict(module.name, open_dim)
                    module.open_cluster(open_dim, self.group[module.name])
                    self.group_status[module.name][open_dim] = 1

                    #  close
                    close_dim = torch.where(self.group_status[module.name]==2)[0]
                    weight = module.weight[close_dim]
                    avg = torch.mean(weight, dim=1).unsqueeze(1).detach()
                    loss_set = torch.mean(torch.abs(weight - avg),dim=(1,2))
                    close_dim = close_dim[torch.where(loss_set<self.args.set_lr)]
                    module.close_cluster(close_dim, self.value_total[module.name])
                    self.group[module.name][close_dim] = 0
                    self.group_time[module.name][close_dim] = 0
                    self.group_status[module.name][close_dim] = 0
                    print(set_dim, open_dim, close_dim)

            th.set_printoptions(6, sci_mode=False)
            for (name, module) in self.mac.agent.named_modules():
                if isinstance(module, GradLayer):
                    grads = []
                    for i in range(self.neu_grad[module.name].shape[0]):  # (agent,dim)
                        grad = self.neu_grad[module.name][i].tolist()
                        grad = [float(f"{x:.8f}") for x in grad]
                        grads.append(grad)
                    self.logger.log_stat('grad_%s' % module.name, grads, t_env)

                    values = []
                    for i in range(self.value_total[module.name].shape[0]):
                        value = self.value_total[module.name][i].tolist()
                        value = [float(f"{x:.6f}") for x in value]
                        values.append(value)
                    self.logger.log_stat('value_%s' % module.name, values, t_env)

                    # print("neural_grad_%s %d %d" % (module.name, i, t1_env), value)
                    self.grad_seq[module.name] = torch.cat((self.grad_seq[module.name][1:], self.neu_grad[module.name].unsqueeze(0)), dim=0)  # (t, agent, dim)
                    efficiency = torch.abs(torch.sum(self.neu_grad[module.name], dim=0)) / torch.sum(torch.abs(self.neu_grad[module.name]),dim=0)
                    efficiency[torch.isnan(efficiency)] = 1.0
                    self.neu_effi[module.name] = torch.cat((self.neu_effi[module.name][1:], efficiency.unsqueeze(0)), dim=0)  # (t, dim)

                    # effi = torch.mean(self.neu_effi[module.name],dim=0)
                    # print(torch.where(effi<torch.mean(effi)))
                    self.logger.log_stat('group_%s' % module.name, self.group[module.name].tolist(), t_env)
                    self.neu_grad[module.name] = th.zeros_like(self.neu_grad[module.name], device=self.args.device)
                    self.value_total[module.name] = th.zeros_like(self.value_total[module.name], device=self.args.device)

            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def calc_conflict(self, name, dim_list):
        value = self.grad_seq[name]
        for dim in dim_list:
            item = torch.zeros((value.shape[1], value.shape[1]))
            for agent_i in range(0, value.shape[1]):
                for agent_j in range(agent_i + 1, value.shape[1]):
                    sub = torch.abs(value[:, agent_i, dim]) + torch.abs(value[:, agent_j, dim]) - torch.abs(
                        value[:, agent_i, dim] + value[:, agent_j, dim])
                    item[agent_i, agent_j] = torch.sum(sub)
                    item[agent_j, agent_i] = item[agent_i, agent_j]
                    item[torch.isnan(item)] = 1.0
            # th.set_printoptions(2, sci_mode=False)
            # print(item)
            self.group[name][dim] = th.tensor(spectral(value.shape[1], min(self.args.n_agents,self.args.cluster_k), item), device=self.args.device)
            # print(dim, item)
            # print(dim, self.group[name][dim])


    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda(self.args.device)
            self.target_mixer.cuda(self.args.device)

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
