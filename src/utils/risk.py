"""
adapted and modified based on  https://github.com/xtma/dsac
"""
import copy

import numpy as np
import torch
import math

def get_risk_q_mode(risk_type, risk_param, rnd_01, z_values, mode="normal"):
    if mode == "normal":
        return get_risk_q(risk_type, risk_param, rnd_01, z_values)
    else:
        risk_q_value, tau_hat = get_risk_q(risk_type, risk_param, rnd_01, z_values)
        quantile_dim = -1
        action_dim = -2
        agent_dim = -3
        if mode.find("0_VaR")>=0:  #这个是用来消融用的, 20230512
            risk_par = float(mode[mode.find("0_VaR") + len("0_VaR") + 1:])
            # t_rnd01 = torch.ones_like(rnd_01).cuda() * (1 - (1 / 2 * rnd_01.shape[-1]))
            t_risk_q_value, t_tau_hat = get_risk_q("VaR", risk_par, rnd_01, z_values)
            if len(rnd_01.shape) == 4:
                risk_q_value[:, :, 0] = t_risk_q_value[:, :, 0]
                tau_hat[:, :, 0] = t_tau_hat[:, :, 0]
            if len(rnd_01.shape) == 2:
                risk_q_value[0] = t_risk_q_value[0]
                tau_hat[0] = t_tau_hat[0]
            return risk_q_value, tau_hat


"""
rnd_01 是0到1之间的随机数, z_values (mac_out)是对应这个rnd_quantiles分位的分位数的取值。
rnd_01 是riskq_agent 自身会随机产生的0到1之间的数值，但是按照DSAC代码里面的意思，这个要经过转换才变成分位数，这里采纳DSAC的方法，对之前DFAC里面iqn-rnn-agent的quantiles(rnd_01)进行修改
如果是从learner这边调用的话，那么
"""
# 这里要做的事情，就是基于rnd_01,来算risk_q，然后得到真正的risk             rnd_01.shape == (batch.batch_size, episode_length, n_quantile_groups, self.n_quantiles)
# mac_out.shape == (batch.batch_size, episode_length + 1, self.args.n_agents, self.args.n_actions, self.n_quantiles)
def get_risk_q(risk_type, risk_param, rnd_01, z_values):
    tau = None
    if len(rnd_01.shape) == 4 and len(z_values.shape) == 5:#this should be called by learner
        batch_size = z_values.shape[0]
        episode_length = z_values.shape[1]
        # n_quantiles_groups = rnd_01.shape[2]
        n_quantiles = z_values.shape[4]
        n_agents = z_values.shape[2]
        n_actions = z_values.shape[3]
        tau_tmp = rnd_01.expand(-1, -1, n_agents, -1)
        tau_tmp = tau_tmp.unsqueeze(3).expand(-1, -1, -1, n_actions, -1)
        tau, tau_hat, presum_tau = get_tau_hat(tau_tmp)
        risk_q_value = get_risk_q_value(risk_type, risk_param, tau_hat, presum_tau, z_values)
        del tau_tmp
    elif len(rnd_01.shape) == 2 and len(z_values.shape) ==3: #it should be called by the risk_controller
        n_agents = z_values.shape[0]
        n_actions = z_values.shape[1]
        tau_tmp = rnd_01.expand(n_agents, -1, -1)
        tau_tmp = tau_tmp.expand(-1,n_actions,-1)
        tau, tau_hat, presum_tau = get_tau_hat2(tau_tmp)
        risk_q_value = get_risk_q_value(risk_type, risk_param, tau_hat, presum_tau, z_values)
    else:
        print("len(rnd_01.shape)", len(rnd_01.shape), "len(z_values.shape)", len(z_values.shape))
        print("how can you reach here")
    del tau, presum_tau
    return risk_q_value, tau_hat

def normal_cdf(value, loc=0., scale=1.):
    return 0.5 * (1 + torch.erf((value - loc) / scale / np.sqrt(2)))


def normal_icdf(value, loc=0., scale=1.):
    return loc + scale * torch.erfinv(2 * value - 1) * np.sqrt(2)


def normal_pdf(value, loc=0., scale=1.):
    return torch.exp(-(value - loc)**2 / (2 * scale**2)) / scale / np.sqrt(2 * np.pi)


"""
#assume that n_quantiles is the last dim
rnd_01 is a random variable between 0 and 1, shape [batch_size, n_agents, n_actions, n_quantiles]
"""
def get_tau_hat(rnd_01):
    presum_tau = rnd_01.clone() #spl的rnd_01，只需要生成一堆间隔，比如n_support=10的话，rnd_01就固定为1/10就行了
    presum_tau /= presum_tau.sum(dim=-1, keepdims=True)  #这个代表了两个分位数之间的距离，也就是DSAC论文里面第7页的\tau_{i+1} - \tau_{i}
    tau = torch.cumsum(presum_tau, dim=-1)
    with torch.no_grad():
        tau_hat = torch.zeros_like(tau).cuda()
        tau_hat[:,:,:,:, 0:1] = tau[:,:,:,:, 0:1] / 2.
        tau_hat[:,:,:,:, 1:] = (tau[:,:,:,:, 1:] + tau[:,:,:,:, :-1]) / 2. #两个分位数之间的中间分位数，这个是真实逼近的时候，求得的
    return tau, tau_hat, presum_tau


def get_tau_hat3(rnd_01):
    presum_tau = rnd_01.clone()
    presum_tau /= presum_tau.sum(dim=-1, keepdims=True)  #这个代表了两个分位数之间的距离，也就是DSAC论文里面第7页的\tau_{i+1} - \tau_{i}
    tau = torch.cumsum(presum_tau, dim=-1)
    with torch.no_grad():
        tau_hat = torch.zeros_like(tau).cuda()
        tau_hat[:,:,:, 0:1] = tau[:,:,:, 0:1] / 2.
        tau_hat[:,:,:, 1:] = (tau[:,:,:, 1:] + tau[:,:,:, :-1]) / 2. #两个分位数之间的中间分位数，这个是真实逼近的时候，求得的
    return tau, tau_hat, presum_tau

def get_tau_hat2(rnd_01):
    presum_tau = rnd_01.clone()
    presum_tau /= presum_tau.sum(dim=-1, keepdims=True)  #这个代表了两个分位数之间的距离，也就是DSAC论文里面第7页的\tau_{i+1} - \tau_{i}
    tau = torch.cumsum(presum_tau, dim=-1)
    with torch.no_grad():
        tau_hat = torch.zeros_like(tau).cuda()
        tau_hat[:,:, 0:1] = tau[:,:, 0:1] / 2.
        tau_hat[:,:, 1:] = (tau[:,:, 1:] + tau[:,:, :-1]) / 2. #两个分位数之间的中间分位数，这个是真实逼近的时候，求得的
    return tau, tau_hat, presum_tau


"""
这个是给agent调用的。在原有iqn_rnn_agent的代码里面会生成随机数（quantiles)，然后agents会返回，为了使得代码和dsac代码的兼容性，必须将quantiles输入来得到tau_hat，之后进行处理，并且需要agent还是返回quantiles
"""
def get_tau_hat_agent(rnd_01):
    presum_tau = rnd_01.clone()
    presum_tau /= presum_tau.sum(dim=-1, keepdims=True)  #这个代表了两个分位数之间的距离，也就是DSAC论文里面第7页的\tau_{i+1} - \tau_{i}
    tau = torch.cumsum(presum_tau, dim=-1)
    with torch.no_grad():
        tau_hat = torch.zeros_like(tau).cuda()
        tau_hat[:,0:1] = tau[:, 0:1] / 2.
        tau_hat[:, 1:] = (tau[:, 1:] + tau[:, :-1]) / 2. #两个分位数之间的中间分位数，这个是真实逼近的时候，求得的
    return tau, tau_hat, presum_tau

"""
@:param risk_type \psi such as cvar, wang, cpw, neutral
@:param risk_param \alpha in this paper
"""
def get_risk_q_value(risk_type, risk_param, tau_hat, presum_tau, quantile_values):
    if risk_type == "VaR":
        n_quantiles = quantile_values.shape[-1]
        idx = -2
        if risk_param - 1.0/(n_quantiles * 2) >= 0:
            idx = math.floor((risk_param - 1.0/(n_quantiles * 2)) / (1.0/n_quantiles))
        with torch.no_grad():
            mask = torch.zeros_like(tau_hat)
        mask[...,idx] = 1
        # mask = mask[...,None]
        risk_q_value = torch.sum(mask * quantile_values, dim=-1, keepdims=True)
        return risk_q_value
    else:
        with torch.no_grad():
            risk_weights = distortion_de(tau_hat, risk_type, risk_param)
        risk_q_value = torch.sum(risk_weights * presum_tau * quantile_values, dim=-1, keepdims=True)
    return risk_q_value

def get_current_risk_param(t_env, stop_t, original_param, target_param):
    t = (t_env - t_env % 10000)/10000
    anneal_step = (original_param - target_param) /(stop_t/10000)
    current_param = original_param - anneal_step * t;
    if t_env >= stop_t:
        return target_param
    return current_param



def distortion_fn(tau, mode="neutral", param=0.):
    # Risk distortion function
    tau = tau.clamp(0., 1.)
    if param >= 0:
        if mode == "neutral":
            tau_ = tau
        elif mode == "wang":
            tau_ = normal_cdf(normal_icdf(tau) + param)
        elif mode == "cvar":
            tau_ = (1. / param) * tau
        elif mode == "cpw":
            tau_ = tau**param / (tau**param + (1. - tau)**param)**(1. / param)
        return tau_.clamp(0., 1.)
    else:
        return 1 - distortion_fn(1 - tau, mode, -param)


def distortion_de(tau, mode="neutral", param=0., eps=1e-8):
    # Derivative of Risk distortion function
    tau = tau.clamp(0., 1.)
    if param >= 0:
        if mode == "neutral":
            tau_ = torch.ones_like(tau)
        elif mode == "wang":
            tau_ = normal_pdf(normal_icdf(tau) + param) / (normal_pdf(normal_icdf(tau)) + eps)
        elif mode == "cvar":
            tau_ = (1. / param) * (tau < param)
        elif mode == "cpw":
            g = tau**param
            h = (tau**param + (1 - tau)**param)**(1 / param)
            g_ = param * tau**(param - 1)
            h_ = (tau**param + (1 - tau)**param)**(1 / param - 1) * (tau**(param - 1) - (1 - tau)**(param - 1))
            tau_ = (g_ * h - g * h_) / (h**2 + eps)
        return tau_.clamp(0., 5.)

    else:
        return distortion_de(1 - tau, mode, -param)
