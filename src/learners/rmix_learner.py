import copy
import numpy as np
from torch import nn

from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import Adam, RMSprop
from modules.mixers.rmix import RMixer
from utils.torch_utils import huber
from modules.layers.grad_layer import GradLayer
from torch.nn import Linear,GRU


def try_detach(maybe_tensor):
    if isinstance(maybe_tensor, int) or isinstance(maybe_tensor, float):
        return maybe_tensor
    else:
        return maybe_tensor.detach().cpu().numpy()


class RMIXLearner:
    """The learner of RMIX"""
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            self.mixer = RMixer(args)

            self.params += list(self.mixer.parameters())
            if "no_mix" in args.name or "none" in args.name:
                self.target_mixer = self.mixer
            else:
                self.target_mixer = copy.deepcopy(self.mixer)

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

        self.cvar_optimiser = Adam(params=self.params, lr=args.lr)
        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        if "no_agent" in args.name or "none" in args.name:
            self.target_mac = mac
        else:
            self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.cumulative_density = th.tensor(
            (2 * np.arange(self.args.num_atoms) + 1) / (2.0 * self.args.num_atoms), dtype=th.float32).view(1, -1).to(self.args.device)
        
        self._temp_qr_loss = th.tensor(0, device=self.args.device)
        self._temp_qr_td = th.tensor(0, device=self.args.device)
        # use to update the distribution without update the CVaR loss
        # not all params need to be updated
        self.last_qr_update_t = 0
        self.dist_optimiser = Adam(params=self.params, lr=args.qr_lr, eps=0.01/32)

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
        mac_out, mac_logits_out, masks, mac_risk_levels = [], [], [], []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs, logits = self.mac.forward(batch, t=t)
            masks.append(self.mac.mask.view(self.args.batch_size, self.args.n_agents, -1))
            mac_out.append(agent_outs)
            mac_risk_levels.append(self.mac.current_risk_level)

            if self.update_qr(t_env):
                mac_logits_out.append(logits)
        
        mac_out = th.stack(mac_out, dim=1)  # Concat over time
        if not self.args.alpha_risk_static:
            mac_risk_levels = th.stack(mac_risk_levels, dim=1)  # Concat over time
        if self.update_qr(t_env):
            mac_logits_out = th.stack(mac_logits_out, dim=1)  # Concat over time
        masks = th.stack(masks, dim=1)
        
        _chosen_action_qdist = None
        # Pick the Q-Values and distribution for the actions taken by each agent
        _chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        if self.update_qr(t_env):
            _actions = actions[..., None].repeat(*([1]*len(actions.size())), self.args.num_atoms)
            _chosen_action_qdist = th.gather(mac_logits_out[:, :-1], dim=3, index=_actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out, target_logits_out, target_mask, target_mac_risk_levels = [], [], [], []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs, target_logits = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
            target_mac_risk_levels.append(self.target_mac.current_risk_level)
            if self.update_qr(t_env):
                target_logits_out.append(target_logits)
            target_mask.append(self.target_mac.mask)

        # We don't need the first (when n_step=1) timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[self.args.n_step:], dim=1)  # Concat across time
        if not self.args.alpha_risk_static:
            target_mac_risk_levels = th.stack(target_mac_risk_levels[self.args.n_step:], dim=1)   # Concat across time
        if self.update_qr(t_env):
            target_logits_out = th.stack(target_logits_out[self.args.n_step:], dim=1)  # Concat across time
        target_masks = th.stack(target_mask, dim=1)

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999
        if self.update_qr(t_env):
            _avail_actions = avail_actions[..., None].repeat(*([1]*len(avail_actions.size())), self.args.num_atoms)
            target_logits_out[_avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        _target_max_qdist = None
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            _target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            if self.update_qr(t_env):
                _cur_max_actions = cur_max_actions[..., None].repeat(*([1]*len(cur_max_actions.size())), self.args.num_atoms)
                _target_max_qdist = th.gather(target_logits_out, 3, _cur_max_actions).squeeze(3)
        else:
            _target_max_qvals = target_mac_out.max(dim=3)[0]
            cur_max_actions = target_mac_out[:, 1:].max(dim=3, keepdim=True)[1]
            
            if self.update_qr(t_env):
                _cur_max_actions = cur_max_actions[..., None].repeat(*([1]*len(cur_max_actions.size())), self.args.num_atoms)
                _target_max_qdist = th.gather(target_logits_out, 3, _cur_max_actions).squeeze(3)

        # Mix
        if self.mixer is not None:
            chosen_action_qvals, chosen_action_qdist = self.mixer(_chosen_action_qvals, 
                                                                  _chosen_action_qdist, 
                                                                  batch["state"][:, :-1],
                                                                  masks[:, :-1],
                                                                  rewards)
            target_max_qvals, target_max_qdist = self.target_mixer(_target_max_qvals, 
                                                                   _target_max_qdist, 
                                                                   batch["state"][:, 1:],
                                                                   target_masks[:, :-1],
                                                                   rewards)
        else:
            chosen_action_qvals, chosen_action_qdist = _chosen_action_qvals, _chosen_action_qdist
            target_max_qvals, target_max_qdist = _target_max_qvals, _target_max_qdist

        # CVaR loss
        # Calculate 1-step Q-Learning targets
        cvar_targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # CVaR Td-error
        cvar_td_error = (chosen_action_qvals - cvar_targets.detach())

        cvar_mask = mask.expand_as(cvar_td_error)

        # 0-out the targets that came from padded data
        masked_cvar_td_error = cvar_td_error * cvar_mask

        # Normal L2 loss, take mean over actual data
        cvar_loss = (masked_cvar_td_error ** 2).sum() / cvar_mask.sum()

        if self.update_qr(t_env):
            # Quantile target here
            # Calculate 1-step Q-Learning targets
            qr_targets = _chosen_action_qvals.detach()[..., None] + self.args.gamma * (1 - terminated[..., None]) * _target_max_qdist

            # QR Td-error
            qr_td_error = (_chosen_action_qdist[..., None] - qr_targets[:, :, :, None, :].detach())
            # 0-out the targets that came from padded data
            masked_qr_td_error = qr_td_error * mask[..., None, None]
            
            # Quantile Huber Loss
            qr_loss = huber(masked_qr_td_error) * (self.cumulative_density - (masked_qr_td_error.detach() < 0).float()).abs()
           
            qr_loss = qr_loss.sum(-1).mean(3).sum() / mask.repeat(1, 1, 3).sum()
            
            loss = cvar_loss
            self._temp_qr_loss = qr_loss
            self._temp_qr_td = masked_qr_td_error
        else:
            qr_loss = self._temp_qr_loss
            masked_qr_td_error = self._temp_qr_td

            loss = cvar_loss

        if check:
            self.find(t_env)
            self.cvar_optimiser.zero_grad()
            self.dist_optimiser.zero_grad()
            return

        if self.update_qr(t_env):
            self.dist_optimiser.zero_grad()
            qr_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
            self.dist_optimiser.step()
        else:
            # Optimise cvar loss
            self.cvar_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)

            for (name, module) in self.mac.agent.named_modules():
                if isinstance(module, GradLayer):
                    self.neu_grad[module.name] += module.weight.grad.detach()
                    self.value_total[module.name] += self.neu_value[module.name].detach()
                    module.weight.grad = None

            self.cvar_optimiser.step()

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
            self.logger.log_stat("cvar_loss", cvar_loss.item(), t_env)
            self.logger.log_stat("qr_loss", qr_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("cvar_td_error_abs", (masked_cvar_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("qr_td_error_abs", (masked_qr_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (cvar_targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            # self.logger.log_stat("batch_rewards_mean", (rewards * mask).sum().item()/mask_elems, t_env)
            # self.logger.log_stat("batch_rewards", rewards.cpu().numpy(), t_env)
            
            # self.logger.log_stat("cvar_chosen_action_qvals", chosen_action_qvals.detach().cpu().numpy(), t_env)
            # self.logger.log_stat("cvar_target_max_qvals", target_max_qvals.detach().cpu().numpy(), t_env)
            # QR distributions
            if self.update_qr(t_env):
                pass
                # self.logger.log_stat("qr_orig_target_max_qdist", _target_max_qdist.detach().cpu().numpy(), t_env)
                # self.logger.log_stat("qr_mixer_target_max_qdist", target_max_qdist.detach().cpu().numpy(), t_env)
                # self.logger.log_stat("qr_orig_chosen_action_qdist", _chosen_action_qdist.detach().cpu().numpy(), t_env)
                # self.logger.log_stat("qr_mixer_chosen_action_qdist", chosen_action_qdist.detach().cpu().numpy(), t_env)

            # save distribution and CVaR and Risk level
            if self.args.version in {'v1', 'v2'}:
                if self.args.save_risk_level:
                    pass
                    # self.logger.log_stat("risk_level", try_detach(self.mac.agent.current_risk_level), t_env)

            self.log_stats_t = t_env
        
        # fianally update the self.last_qr_update_t
        if self.update_qr(t_env):
            self.last_qr_update_t = t_env

    def update_qr(self, t_env):
        if (t_env - self.last_qr_update_t) / self.args.qr_update_interval >= 1.0 and t_env > self.args.start_to_update_qr:
            return True
        else:
            return False

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
