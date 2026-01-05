import torch
import torch.nn as nn
from torch.nn import init
import os
import json
from .Swin import SwinTransformer
from typing import Tuple


class MultiView(nn.Module):
    def __init__(self, config, logger, data_mean=None, data_range=None):
        super(MultiView, self).__init__()

        self.config = config
        self.config_train = config['train']
        self.logger = logger
        self.is_train = config['is_train']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_norm = config['train']['data_norm']
        self.data_mean = data_mean
        self.data_range = data_range

        self.net = SwinTransformer(
            img_size=config['net']['img_size'],
            patch_size=config['datasets']['patch_size'],
            in_chans=config['net']['in_chans'],
            embed_dim=config['net']['embed_dim'],
            depth=config['net']['depth'],
            num_heads=config['net']['num_heads'],
            window_size=config['net']['window_size'],
            mlp_ratio=config['net']['mlp_ratio'],
            qkv_bias=config['net']['qkv_bias'],
            drop_rate=config['net']['drop_rate'],
            attn_drop_rate=config['net']['attn_drop_rate'],
            resi_connection=config['net']['resi_connection'])
        self.net.apply(self.weight_init)

        loss_type = config['train']['loss_type']
        dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        if loss_type == 'l1':
            self.loss = nn.L1Loss().type(dtype).to(self.device)
        elif loss_type == 'l2':
            self.loss = nn.MSELoss().type(dtype).to(self.device)
        self.loss_weight = config['train']['loss_weight']

        self.load_model()

    def weight_init(self, m, init_type='xavier_normal', gain=1):

        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        #
        #     if init_type == 'normal':
        #         init.normal_(m.weight.data, 0, 0.1)
        #         m.weight.data.clamp_(-1, 1).mul_(gain)
        #
        #     elif init_type == 'uniform':
        #         init.uniform_(m.weight.data, -0.2, 0.2)
        #         m.weight.data.mul_(gain)
        #
        #     elif init_type == 'xavier_normal':
        #         init.xavier_normal_(m.weight.data, gain=gain)
        #         m.weight.data.clamp_(-1, 1)
        #
        #     elif init_type == 'xavier_uniform':
        #         init.xavier_uniform_(m.weight.data, gain=gain)
        #
        #     elif init_type == 'kaiming_normal':
        #         init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')#, nonlinearity='relu')
        #         #m.weight.data.clamp_(-1, 1).mul_(gain)
        #
        #     elif init_type == 'kaiming_uniform':
        #         init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        #         m.weight.data.mul_(gain)
        #
        #     elif init_type == 'orthogonal':
        #         init.orthogonal_(m.weight.data, gain=gain)
        #
        #     else:
        #         raise NotImplementedError('Initialization method [{:s}] is not implemented'.format(init_type))
        #
        #     if m.bias is not None:
        #         m.bias.data.zero_()

    def load_model(self, strict=True, param_key='params'):
        load_path = self.config['path']['pretrain']
        if load_path is not None:
            self.logger.info("Load from pretrained dir: {}".format(load_path))
            if strict:
                state_dict = torch.load(load_path)
                if param_key in state_dict.keys():
                    state_dict = state_dict[param_key]
                self.net.load_state_dict(state_dict, strict=strict)
            else:
                state_dict_old = torch.load(load_path)
                if param_key in state_dict_old.keys():
                    state_dict_old = state_dict_old[param_key]
                state_dict = self.net.state_dict()
                for ((key, val), (key_old, val_old)) in zip(state_dict.items(), state_dict_old.items()):
                    state_dict[key] = val_old
                self.net.load_state_dict(state_dict, strict=strict)
                del state_dict, state_dict_old

    def forward(self, x):
        if self.data_norm:
            x = (x - self.data_mean) / self.data_range  # normalize to [-1, 1]
        out = self.net(x)
        if self.data_norm:
            out = out * self.data_range + self.data_mean
        return out

    def rec_loss(self, x, y, clean=None, sum_reg=False):
        out = self.forward(x)
        loss = self.loss(out, y)
        if sum_reg:
            # square error between sum of clean and sum of out
            sum_loss = 0.02 * self.loss(torch.sum(out), torch.sum(clean)) / self.config['net']['img_size'] ** 2
            #loss += sum_loss
            return loss, sum_loss

        return loss

    def save_model(self, res_str, save_path):
        if not self.config['is_train']:
            return
        save_name = save_path + '/{}_img{}_w{}_h{}_e{}_d{}'.format(self.config['train']['model'],
                                                                   self.config['net']['img_size'],
                                                                   self.config['net']['window_size'],
                                                                   self.config['net']['num_heads'],
                                                                   self.config['net']['embed_dim'],
                                                                   self.config['net']['drop_rate']
                                                                   )
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        torch.save(self.net.state_dict(), save_name + '.pth')

        res = self.config
        res['result'] = res_str
        # save config
        with open(save_name + '.json', 'w') as f:
            json.dump(res, f)


if __name__ == "__main__":
    import json

    config = json.load(open('./config.json'))
    config['is_train'] = True
    config['net']['window_size'] = 3
    config['net']['img_size'] = 3

    import numpy as np
    import torch

    model = MultiView(config, None).cuda()
    x = np.array([[[0, 0, 0], [0, 20, 17], [0, 5, 8]]])
    x = torch.tensor(x).float().cuda()
    x = x.unsqueeze(0)

    out = model(x)
    print(out)
