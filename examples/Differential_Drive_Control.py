#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
#from IPython import get_ipython
from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods
from mpc.env_dx import differential_drive

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

import os
import io
import base64
import tempfile
#from IPython.display import HTML

from tqdm import tqdm



# In[2]:


params = torch.tensor((10., 1., 1.))

dx = differential_drive.DiffDriveDx(params, simple=True)

n_batch, T, mpc_T = 1, 100, 60



xinit = torch.stack((torch.tensor([0.0]), torch.tensor([0.0]),torch.tensor([0.0]),torch.tensor([0.0]),torch.tensor([0.0])), dim=1)
torch.tensor(0.0)
x = xinit
u_init = torch.stack((torch.tensor([0.0]),(torch.tensor([0.0]))), dim=1)



mode = 'reference-following'
# mode = 'spin'

if mode == 'reference-following':
    goal_weights = torch.Tensor((30.0, 30.0, 3.0, 0.1,0.01))
    goal_state = torch.Tensor((1., 1., 0. , 0. , 0.))
    ctrl_penalty = torch.Tensor((0.0001, 0.0001))
    #ctrl_penalty = 0.00001

    q = torch.cat((
        goal_weights,
        ctrl_penalty
    ))
    #px = -torch.sqrt(goal_weights)*goal_state
    px = -goal_weights * goal_state
    p = torch.cat((px, torch.zeros(dx.n_ctrl)))
    p = p.unsqueeze(0).repeat(mpc_T, n_batch, 1)

    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
        mpc_T, n_batch, 1, 1
    )



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
    print(next_action)
    u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, dx.n_ctrl)), dim=0)
    x = dx(x, next_action)

    n_row, n_col = 1, 2
    fig, axs = plt.subplots(n_row, n_col, figsize=(3 * n_col,3 * n_row))
    axs = axs.reshape(-1)
    for i in range(n_batch):
        dx.get_frame(x[i],next_action, ax=axs[i])

    fig.tight_layout()
    fig.savefig(os.path.join(t_dir, '{:03d}.png'.format(t)))
    plt.close(fig)


# In[3]:


vid_fname = 'diff_drive-{}.mp4'.format(mode)

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
#HTML(data='''<video alt="test" controls>
#                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
#             </video>'''.format(encoded.decode('ascii')))

