#!/usr/bin/env python
# coding: utf-8
import gc
import time

# In[1]:


import torch
# from IPython import get_ipython
from mpc import mpc
from mpc.mpc import QuadCost, LinDx, GradMethods
from mpc.env_dx import pendulum

import numpy as np
import numpy.random as npr

import matplotlib.pyplot as plt

import os
import io
import base64
import tempfile
from IPython.display import HTML

from tqdm import tqdm

from mpc.util import print_torch_memory_allocated

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

# PARAMS:
debug_memory_mode = False
# n_batch, T, mpc_T = 5000, 100, 5
n_batch, T, mpc_horizon = 1_000_000, 100, 10
generate_video = False

params = torch.tensor((10., 1., 1.), device=device)  # Not sure what these params are
dx = pendulum.PendulumDx(params, simple=True, debug_memory_mode=debug_memory_mode)
dx.to(device)
if debug_memory_mode:
    params.debug_name = f"(PDx) params"

gc.enable()


def garbage_collection_cuda() -> None:
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    try:
        # This is the last thing that should cause an OOM error, but seemingly it can.
        torch.cuda.empty_cache()
    except RuntimeError as exception:
        pass


def print_gpu_summary():
    summary = torch.cuda.memory_summary(device=device, abbreviated=True)
    print(summary)


def uniform(shape, low, high):
    r = high - low
    return torch.rand(shape, device=device) * r + low


torch.manual_seed(0)
th = uniform(n_batch, -(1 / 2) * torch.pi, (1 / 2) * torch.pi)
thdot = uniform(n_batch, -1., 1.)
xinit = torch.stack((torch.cos(th), torch.sin(th), thdot), dim=1)

x = xinit
u_init = None

if debug_memory_mode:
    xinit.debug_name = f"(PC) xinit"
    x.debug_name = f"(PC) x"
    th.debug_name = f"(PC) th"
    thdot.debug_name = f"(PC) thdot"

# The cost terms for the swingup task can be alternatively obtained
# for this pendulum environment with:
# q, p = dx.get_true_obj()

mode = 'swingup'
# mode = 'spin'

if mode == 'swingup':
    goal_weights = torch.Tensor([1., 1., 0.1]).to(device)
    goal_state = torch.Tensor([1., 0., 0.]).to(device)
    ctrl_penalty = 0.001
    q = torch.cat((
        goal_weights,
        ctrl_penalty * torch.ones(dx.n_ctrl, device=device)
    ))
    px = -torch.sqrt(goal_weights) * goal_state
    p = torch.cat((px, torch.zeros(dx.n_ctrl, device=device)))
    Q = torch.diag(q).unsqueeze(0).unsqueeze(0).repeat(
        mpc_horizon, n_batch, 1, 1
    )
    p = p.unsqueeze(0).repeat(mpc_horizon, n_batch, 1)
    if debug_memory_mode:
        goal_weights.debug_name = f"(PC) goal_weights"
        goal_state.debug_name = f"(PC) goal_state"
        q.debug_name = f"(PC) q"
        px.debug_name = f"(PC) px"
        p.debug_name = f"(PC) p"
        Q.debug_name = f"(PC) Q"

elif mode == 'spin':
    Q = 0.001 * torch.eye(dx.n_state + dx.n_ctrl).unsqueeze(0).unsqueeze(0).repeat(
        mpc_horizon, n_batch, 1, 1
    )
    p = torch.tensor((0., 0., -1., 0.))
    if debug_memory_mode:
        p.debug_name = f"(PC) p (before mods)"
    p = p.unsqueeze(0).repeat(mpc_horizon, n_batch, 1)
    if debug_memory_mode:
        Q.debug_name = f"(PC) Q"
        p.debug_name = f"(PC) p"

t_dir = tempfile.mkdtemp()
print('Tmp dir: {}'.format(t_dir))
my_mpc = mpc.MPC(
    dx.n_state, dx.n_ctrl, mpc_horizon,
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
    debug_memory_mode=debug_memory_mode,
)
my_mpc.to(device)
if debug_memory_mode:
    my_mpc.debug_name = f"(PC) my_mpc"
for t in tqdm(range(T)):
    t1 = time.perf_counter()
    if debug_memory_mode:
        print_torch_memory_allocated("(Before MPC call)",print_tensors=True)
    nominal_states, nominal_actions, nominal_objs = my_mpc.forward(x, QuadCost(Q, p), dx)
    if debug_memory_mode:
        nominal_states.debug_name = f"(PC) nominal_states"
        nominal_actions.debug_name = f"(PC) nominal_actions"
        nominal_objs.debug_name = f"(PC) nominal_objs"
    # my_mpc.zero_grad()
    if debug_memory_mode:
        print_torch_memory_allocated("(out of scope test)", print_tensors=True)
    garbage_collection_cuda()
    t2 = time.perf_counter()
    # print('MPC took {:.03f} sec'.format(t2 - t1))
    next_action = nominal_actions[0]
    u_init = torch.cat((nominal_actions[1:], torch.zeros(1, n_batch, dx.n_ctrl, device=device)), dim=0)
    u_init[-2] = u_init[-3]
    x = dx(x, next_action)
    if debug_memory_mode:
        next_action.debug_name = f"(PC) next_action"
        u_init.debug_name = f"(PC) u_init"
        x.debug_name = f"(PC) x"
    del nominal_states, nominal_actions, nominal_objs

    if generate_video:
        n_row, n_col = 4, 4
        fig, axs = plt.subplots(n_row, n_col, figsize=(3 * n_col, 3 * n_row))
        axs = axs.reshape(-1)
        for i in range(n_batch):
            dx.get_frame(x[i], ax=axs[i])
            axs[i].get_xaxis().set_visible(False)
            axs[i].get_yaxis().set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(t_dir, '{:03d}.png'.format(t)))
        plt.close(fig)

if generate_video:
    vid_fname = 'pendulum-{}.mp4'.format(mode)

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
