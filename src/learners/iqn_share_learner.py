import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.ddn import DDNMixer
from modules.mixers.dmix import DMixer
import torch as th
import torch
import torch.nn.functional as F
from torch.optim import RMSprop
from torch.optim import Adam, SGD
import numpy as np
from modules.layers.grad_layer import GradLayer
from modules.layers.share_layer import ShareLayer
from utils.cluster import spectral

class IQNShareLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "ddn":
                self.mixer = DDNMixer(args)
            elif args.mixer == "dmix":
                self.mixer = DMixer(args)
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
                    # return None
                module.register_forward_hook(hook=forward_hook)
                # module.register_backward_hook(hook=backward_hook)

        if args.optimizer == "RMSProp":
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)
        elif args.optimizer == "Adam":
            self.optimiser = Adam(params=self.params, lr=args.lr, eps=args.optim_eps)
        else:
            raise ValueError("Unknown Optimizer")

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC

        self.target_mac = copy.deepcopy(mac)
        self.set_optimiser = SGD(params=self.params, lr=args.set_lr)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_quantiles = args.n_quantiles
        self.n_target_quantiles = args.n_target_quantiles

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        episode_length = rewards.shape[1]
        assert rewards.shape == (batch.batch_size, episode_length, 1)
        actions = batch["actions"][:, :-1]
        assert actions.shape == (batch.batch_size, episode_length, self.args.n_agents, 1)
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        assert mask.shape == (batch.batch_size, episode_length, 1)
        avail_actions = batch["avail_actions"]
        assert avail_actions.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions)

        for key in self.neu_value.keys():
            self.neu_value[key] = th.zeros_like(self.neu_value[key], device=self.args.device)

        # Mix
        if self.mixer is not None:
            # Same quantile for quantile mixture
            n_quantile_groups = 1
        else:
            n_quantile_groups = self.args.n_agents

        # Calculate estimated Q-Values
        mac_out = []
        rnd_quantiles = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, agent_rnd_quantiles = self.mac.forward(batch, t=t, forward_type="policy")
            assert agent_outs.shape == (batch.batch_size * self.args.n_agents, self.args.n_actions, self.n_quantiles)
            assert agent_rnd_quantiles.shape == (batch.batch_size * n_quantile_groups, self.n_quantiles)
            agent_rnd_quantiles = agent_rnd_quantiles.view(batch.batch_size, n_quantile_groups, self.n_quantiles)
            rnd_quantiles.append(agent_rnd_quantiles)
            agent_outs = agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_quantiles)
            mac_out.append(agent_outs)
        del agent_outs
        del agent_rnd_quantiles
        mac_out = th.stack(mac_out, dim=1) # Concat over time
        rnd_quantiles = th.stack(rnd_quantiles, dim=1) # Concat over time
        assert mac_out.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions, self.n_quantiles)
        assert rnd_quantiles.shape == (batch.batch_size, episode_length+1, n_quantile_groups, self.n_quantiles)
        rnd_quantiles = rnd_quantiles[:,:-1]
        assert rnd_quantiles.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)

        # Pick the Q-Values for the actions taken by each agent
        actions_for_quantiles = actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)
        del actions
        chosen_action_qvals = th.gather(mac_out[:,:-1], dim=3, index=actions_for_quantiles).squeeze(3)  # Remove the action dim
        del actions_for_quantiles
        assert chosen_action_qvals.shape == (batch.batch_size, episode_length, self.args.n_agents, self.n_quantiles)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, _ = self.target_mac.forward(batch, t=t, forward_type="target")
            assert target_agent_outs.shape == (batch.batch_size * self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            target_agent_outs = target_agent_outs.view(batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            assert target_agent_outs.shape == (batch.batch_size, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)
            target_mac_out.append(target_agent_outs)
        del target_agent_outs
        del _

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
        assert target_mac_out.shape == (batch.batch_size, episode_length, self.args.n_agents, self.args.n_actions, self.n_target_quantiles)

        # Mask out unavailable actions
        assert avail_actions.shape == (batch.batch_size, episode_length+1, self.args.n_agents, self.args.n_actions)
        target_avail_actions = avail_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        target_mac_out[target_avail_actions[:,1:] == 0] = -9999999
        avail_actions = avail_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_quantiles)

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:,1:].mean(dim=4).max(dim=3, keepdim=True)[1]
            del mac_out_detach
            assert cur_max_actions.shape == (batch.batch_size, episode_length, self.args.n_agents, 1)
            cur_max_actions = cur_max_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            del cur_max_actions
        else:
            # [0] is for max value; [1] is for argmax
            cur_max_actions = target_mac_out.mean(dim=4).max(dim=3, keepdim=True)[1]
            assert cur_max_actions.shape == (batch.batch_size, episode_length, self.args.n_agents, 1)
            cur_max_actions = cur_max_actions.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            del cur_max_actions
        del target_mac_out
        assert target_max_qvals.shape == (batch.batch_size, episode_length, self.args.n_agents, self.n_target_quantiles)

        # Mix
        if self.mixer is not None:
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], target=True)
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1], target=False)
            assert chosen_action_qvals.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)
            assert target_max_qvals.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_target_quantiles)

        # Calculate 1-step Q-Learning targets
        target_samples = rewards.unsqueeze(3) + \
            (self.args.gamma * (1 - terminated)).unsqueeze(3) * \
            target_max_qvals
        del target_max_qvals
        del rewards
        del terminated
        assert target_samples.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_target_quantiles)

        # Quantile Huber loss
        target_samples = target_samples.unsqueeze(3).expand(-1, -1, -1, self.n_quantiles, -1)
        assert target_samples.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        assert chosen_action_qvals.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)
        chosen_action_qvals = chosen_action_qvals.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        assert chosen_action_qvals.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # u is the signed distance matrix
        u = target_samples.detach() - chosen_action_qvals
        del target_samples
        del chosen_action_qvals
        assert u.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        assert rnd_quantiles.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)
        tau = rnd_quantiles.unsqueeze(4).expand(-1, -1, -1, -1, self.n_target_quantiles)
        assert tau.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # The abs term in quantile huber loss
        abs_weight = th.abs(tau - u.le(0.).float())
        del tau
        assert abs_weight.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # Huber loss
        loss = F.smooth_l1_loss(u, th.zeros(u.shape).cuda(self.args.device), reduction='none')
        del u
        assert loss.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles, self.n_target_quantiles)
        # Quantile Huber loss
        loss = (abs_weight * loss).mean(dim=4).sum(dim=3)
        del abs_weight
        assert loss.shape == (batch.batch_size, episode_length, n_quantile_groups)

        assert mask.shape == (batch.batch_size, episode_length, 1)
        mask = mask.expand_as(loss)

        # 0-out the targets that came from padded data
        loss = loss * mask

        loss = loss.sum() / mask.sum()
        assert loss.shape == ()

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
                    # time_var = torch.mean(torch.var(module.weight.data, dim=1),dim=1)/self.group_time[module.name]
                    time_var = torch.mean(torch.var(module.weight.data, dim=1), dim=1) / self.group_time[module.name]
                    time_var[torch.where(self.group_status[module.name] != 1)] = float('inf')
                    self.group_var[module.name] = time_var

                    print(self.group_var[module.name])
                    #  reset
                    # reset = torch.rand(self.group_status[module.name].shape[0], device=self.args.device)
                    # set_dim = torch.where((reset <= 0.00) & (self.group_status[module.name] == 1))[0]
                    reset = torch.rand(1, device=self.args.device)
                    set_dim = -1
                    if reset<self.args.reset_p*torch.count_nonzero(self.group_status[module.name]==1)/self.group_status[module.name].shape[0]:
                        set_dim = torch.argmin(self.group_var[module.name])
                        if self.group_status[module.name][set_dim] == 1:
                            self.group_status[module.name][set_dim] = 2

                    #  open
                    # number = torch.count_nonzero(self.group_status[module.name] == 0)
                    self.group_time[module.name] += 1
                    ready_dim = torch.where((self.group_time[module.name] > self.args.share_T)&(self.group_status[module.name] == 0))[0]
                    effi = torch.mean(self.neu_effi[module.name][:, ready_dim],dim=0)
                    open_dim = ready_dim[torch.where(effi<0.2)[0]]

                    self.group_time[module.name][open_dim] = 0
                    self.calc_conflict(module.name, open_dim)
                    module.open_cluster(open_dim, self.group[module.name])
                    self.group_status[module.name][open_dim] = 1

                    #  close
                    close_dim = torch.where(self.group_status[module.name]==2)[0]
                    weight = module.weight[close_dim]
                    avg = torch.mean(weight, dim=1).unsqueeze(1).detach()
                    loss_set = torch.mean(torch.abs(weight - avg),dim=(1,2))
                    print(loss_set)
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
                    self.grad_seq[module.name] = torch.cat(
                        (self.grad_seq[module.name][1:], self.neu_grad[module.name].unsqueeze(0)),
                        dim=0)  # (t, agent, dim)
                    efficiency = torch.abs(torch.sum(self.neu_grad[module.name], dim=0)) / torch.sum(
                        torch.abs(self.neu_grad[module.name]), dim=0)
                    efficiency[torch.isnan(efficiency)] = 1.0
                    self.neu_effi[module.name] = torch.cat((self.neu_effi[module.name][1:], efficiency.unsqueeze(0)),
                                                           dim=0)  # (t, dim)

                    # effi = torch.mean(self.neu_effi[module.name],dim=0)
                    # print(torch.where(effi<torch.mean(effi)))
                    self.logger.log_stat('group_%s' % module.name, self.group[module.name].tolist(), t_env)
                    self.neu_grad[module.name] = th.zeros_like(self.neu_grad[module.name], device=self.args.device)
                    self.value_total[module.name] = th.zeros_like(self.value_total[module.name],
                                                                  device=self.args.device)
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
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