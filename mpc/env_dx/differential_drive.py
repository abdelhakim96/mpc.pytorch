# differential drive robot model
# by Hakim Amer 24/7/2023


import torch
from torch.autograd import Function, Variable
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter

import numpy as np

from mpc import util

import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('bmh')

class DiffDriveDx(nn.Module): #change
    def __init__(self, params=None, simple=True):
        super().__init__()
        #self.simple = simple
        self.max_ur = 10.0  # max right wheel speed
        self.max_ul = 10.0  # max left wheel speed
        self.dt = 0.05      # MPC sampling time
        self.n_state = 5    # number of states
        self.n_ctrl = 2     # number of control inputs

        # robot params, Linear Inertia (M), Angular Inertia (J), friction (D), Length(L)
        self.params = Variable(torch.Tensor((1., 1., 0., 1.0)))

        #States: [px, py, v, th, dth]
        self.goal_state = torch.Tensor([1., 1., 0., 0., 0.])

        self.goal_weights = torch.Tensor([50., 50., 0.1, 3., 3.])

        #Control: [u_r, u_l]
        self.ctrl_penalty = torch.Tensor([0.001, 0.001])
        self.lower, self.upper = -10., 10.

        self.mpc_eps = 1e-3  #
        self.linesearch_decay = 0.2
        self.max_linesearch_iter = 5

    def forward(self, x, u):
        squeeze = x.ndimension() == 1

        if squeeze:
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)

        assert x.ndimension() == 2
        assert x.shape[0] == u.shape[0]
        assert x.shape[1] == 5
        assert u.shape[1] == 2
        assert u.ndimension() == 2

        if x.is_cuda and not self.params.is_cuda:
            self.params = self.params.cuda()

        pos_x, pos_y, v, th, dth = torch.unbind(x, dim=1)




        pos_x = pos_x + v * torch.cos(th) * self.dt
        pos_y = pos_y + v * torch.sin(th) * self.dt
        v = v + (u[0,0] + u[0,1]) * self.dt
        th = th + dth * self.dt

        dth = dth + ((u[0, 0] -u[0, 1])* self.params[3] ) * self.dt
        state = torch.stack(( pos_x, pos_y, v ,th, dth ), 1)


        if squeeze:
            state = state.squeeze(0)
        return state

    def get_frame(self, x,u, ax=None):
        x = util.get_data_maybe(x.view(-1))
        assert len(x) == 5


        pos_x, pos_y, v, th, dth = torch.unbind(x)
        #u1 = torch.unbind(u)

        u= torch.detach(u)
        u1 = u[0,0]
        u2 = u[0,1]
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,6))
        else:
            fig = ax.get_figure()

        ax.scatter(pos_x,pos_y, color='blue')
        ax.quiver(pos_x, pos_y, torch.cos(th), torch.sin(th))
        #print('x',pos_x)


        #ax.plot((u1,u2), color='red')

        return fig, ax

    def get_true_obj(self):
        q = torch.cat((
            self.goal_weights,
            self.ctrl_penalty*torch.ones(self.n_ctrl)
        ))
        assert not hasattr(self, 'mpc_lin')
        px = -torch.sqrt(self.goal_weights)*self.goal_state #+ self.mpc_lin
        p = torch.cat((px, torch.zeros(self.n_ctrl)))
        return Variable(q), Variable(p)


if __name__ == '__main__':
    dx = DiffDriveDx()
    n_batch, T = 1, 100
    u = torch.zeros(T, n_batch, dx.n_ctrl)
    xinit = torch.zeros(n_batch, dx.n_state)
    xinit[:,:] = 0.
    x = xinit
    for t in range(T):
        x = dx(x, u[t])
        fig, ax = dx.get_frame(x[0])
        fig.savefig('{:03d}.png'.format(t))
        plt.close(fig)

    vid_file = 'diff_drive_vid.mp4'
    if os.path.exists(vid_file):
        os.remove(vid_file)
    cmd = ('(/usr/bin/ffmpeg -loglevel quiet '
            '-r 32 -f image2 -i %03d.png -vcodec '
            'libx264 -crf 25 -pix_fmt yuv420p {}/) &').format(
        vid_file
    )
    os.system(cmd)
    for t in range(T):
        os.remove('{:03d}.png'.format(t))
