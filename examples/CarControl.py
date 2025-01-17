#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from IPython import get_ipython
from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods
from mpc.env_dx import car

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

import os
import io
import base64
import tempfile
from IPython.display import HTML

from tqdm import tqdm



# In[2]:


params = torch.tensor((10., 1., 1.))

dx = car.CarDx(params, simple=True)

n_batch, T, mpc_T = 1, 100, 20

def uniform(shape, low, high):
    r = high-low
    return torch.rand(shape)*r+low

torch.manual_seed(0)
th = uniform(n_batch, -(1/2)*np.pi, (1/2)*np.pi)
thdot = uniform(n_batch, -1., 1.)
xinit = torch.stack((torch.tensor([0.0]), torch.tensor([0.0]),torch.tensor([0.0]),torch.tensor([0.0]),torch.tensor([0.0])), dim=1)
torch.tensor(0.0)
x = xinit
u_init = torch.stack((torch.tensor([0.0]),(torch.tensor([0.0]))), dim=1)

# The cost terms for the swingup task can be alternatively obtained
# for this pendulum environment with:
# q, p = dx.get_true_obj()

mode = 'reference following'
# mode = 'spin'

if mode == 'reference following':
    goal_weights = torch.Tensor((1., 1., 0.1, 0.,0.))
    goal_state = torch.Tensor((1., 1., 0., 0.,0.))
    ctrl_penalty = torch.Tensor((0.0001, 0.0001))
    q = torch.cat((
        goal_weights,
        ctrl_penalty*torch.ones(dx.n_ctrl)
    ))
    px = -torch.sqrt(goal_weights)*goal_state
    p = torch.cat((px, torch.zeros(dx.n_ctrl)))
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
        mpc_T, n_batch, 1, 1
    )
    p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)


t_dir = tempfile.mkdtemp()
print('Tmp dir: {}'.format(t_dir))

for t in tqdm(range(T)):
    nominal_states, nominal_actions, nominal_objs = mpc.MPC(
        dx.n_state, dx.n_ctrl, mpc_T,
        u_init=u_init,
        u_lower=dx.lower, u_upper=dx.upper,
        lqr_iter=50,
        verbose=0,
        exit_unconverged=False,
        detach_unconverged=False,
        linesearch_decay=dx.linesearch_decay,
        max_linesearch_iter=dx.max_linesearch_iter,
        grad_method=GradMethods.AUTO_DIFF,
        eps=1e-2,
    )(x, QuadCost(Q, p), dx)
    
    next_action = nominal_actions[0]
    u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, dx.n_ctrl)), dim=0)
    u_init[-2] = u_init[-3]
    x = dx(x, next_action)

    n_row, n_col = 4, 4
    fig, axs = plt.subplots(n_row, n_col, figsize=(3*n_col,3*n_row))
    axs = axs.reshape(-1)
    for i in range(n_batch):
        dx.get_frame(x[i], ax=axs[i])
        axs[i].get_xaxis().set_visible(False)
        axs[i].get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(t_dir, '{:03d}.png'.format(t)))
    plt.close(fig)


# In[3]:


vid_fname = 'car-{}.mp4'.format(mode)

if os.path.exists(vid_fname):
    os.remove(vid_fname)
    
cmd = 'ffmpeg -r 16 -f image2 -i {}/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p {}'.format(
    t_dir, vid_fname
)
os.system(cmd)
print('Saving video to: {}'.format(vid_fname))


# In[4]:


video = io.open(vid_fname, 'r+b').read()
encoded = base64.b64encode(video)
HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii')))

