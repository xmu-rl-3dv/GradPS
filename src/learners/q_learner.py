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
from torch.nn import Linear
from torch.nn import GRU
from utils.cluster import spectral

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        # self.device = th.device('cuda' if args.use_cuda else 'cpu')
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

        self.turn = 0

        self.neu_grad = {}
        self.value_total = {}
        self.neu_value = {}

        for (name, module) in self.mac.agent.named_modules():  # time need to be added
            if isinstance(module, GradLayer):  # (agent,dim)
                self.neu_value[module.name] = th.zeros_like(module.weight, device=args.device)
                self.value_total[module.name] = th.zeros_like(module.weight, device=args.device)
                self.neu_grad[module.name] = th.zeros_like(module.weight, device=args.device)

                def forward_hook(module, fea_in, fea_out):
                    fea_out = fea_out.reshape(-1, args.n_agents, fea_out.shape[1])  # t(batch*agent, dim)  avg of each time
                    self.neu_value[module.name] = self.neu_value[module.name] + th.mean(fea_out, dim=0)
                    # return None

                # def backward_hook(module, grad_in, grad_out): # (batch*agent, dim) record the gradient
                # print(module.name)
                # print('grad_in: ',grad_in)
                # print('grad_out: ',grad_out)

                # self.neu_grad[module.name] += torch.sum(grad_out[0].reshape(-1,self.args.n_agents, grad_out[0].shape[1]),dim=0)
                # grad = grad_in[0].reshape(-1, self.args.n_agents, grad_out[0].shape[1])
                # for index in range(self.args.n_agents):
                #     if torch.all(grad[index]==0):

                # self.`neu_grad[module.name] += grad_in[1]
                # return None

                module.register_forward_hook(hook=forward_hook)
                # module.register_backward_hook(hook=backward_hook)

        para = sum([numpy.prod(list(p.size())) for p in self.params])
        print(args.name, para * 4 / 1024 / 1024)

        # self.share_params = [param for param in self.params if torch.all(param == 1)]
        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

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

        # Every Agent's gradient
        # q_grad = chosen_action_qvals*chosen_action_qvals.grad.detach()  # (batch, time, agent)
        # for i in range(q_grad.shape[2]):
        #     loss_agent = q_grad[:,:,i].sum()
        #     self.optimiser.zero_grad()
        #     loss_agent.backward(retain_graph=True)
        #     for (name, module) in self.mac.agent.named_modules():
        #         if isinstance(module, GradLayer):
        #             self.neu_grad[module.name][i] += module.weight.grad.detach()*0.5  # (agent, dim)
        #             module.weight.grad = None

        # for key, value in self.neu_grad.items():
        #     print(key, torch.sum(value, dim=0))

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
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

                    # print("neural_grad_%s %d %d" % (module.name, i, t_env), value)
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

    def reset_turn(self, run_turn):
        self.optimiser.zero_grad()
        self.optimiser = RMSprop(params=self.params, lr=self.args.lr, alpha=self.args.optim_alpha,
                                 eps=self.args.optim_eps)
        self.last_target_update_episode = 0
        self.log_stats_t = -self.args.learner_log_interval - 1
        for key in self.neu_value.keys():
            self.neu_grad[key] = th.zeros_like(self.neu_grad[key], device=self.args.device)
            self.value_total[key] = th.zeros_like(self.value_total[key], device=self.args.device)
            self.neu_value[key] = th.zeros_like(self.neu_value[key], device=self.args.device)
        self._update_targets()
        self.turn = run_turn
