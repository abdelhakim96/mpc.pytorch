import logging
import math
import time
import gym
import numpy as np
import torch
import torch.autograd
from gym import wrappers, logger as gym_log
from mpc import mpc

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

# ... (Previous code remains unchanged)

def run_single_pendulum(i, cost, u_init, render, verbose):
    env = gym.make(ENV_NAME).env
    env.reset()

    if downward_start:
        env.state = [np.pi, 1]

    env = wrappers.Monitor(env, f'/tmp/box_ddp_pendulum/pendulum_{i}/', force=True)
    if downward_start:
        env.env.state = [np.pi, 1]

    total_reward = 0

    for j in range(run_iter):
        state = env.state.copy()
        state = torch.tensor(state).view(1, -1)

        command_start = time.perf_counter()

        # Recreate controller using updated u_init (kind of wasteful right?)
        ctrl = mpc.MPC(nx, nu, TIMESTEPS, u_lower=ACTION_LOW, u_upper=ACTION_HIGH, lqr_iter=LQR_ITER,
                       exit_unconverged=False, eps=1e-2,
                       n_batch=N_BATCH, backprop=False, verbose=0, u_init=u_init,
                       grad_method=mpc.GradMethods.AUTO_DIFF)

        # Compute action based on current state, dynamics, and cost
        nominal_states, nominal_actions, nominal_objs = ctrl(state, cost, PendulumDynamics())
        action = nominal_actions[0]  # Take first planned action
        u_init = torch.cat((nominal_actions[1:], torch.zeros(1, N_BATCH, nu)), dim=0)

        elapsed = time.perf_counter() - command_start
        s, r, _, _ = env.step(action.detach().numpy())
        total_reward += r

        if verbose:
            logger.debug("Pendulum %d, Step %d: Action taken: %.4f, Cost received: %.4f, Time taken: %.5fs",
                         i, j, action, -r, elapsed)
        if render:
            env.render()

    logger.info("Pendulum %d, Total reward: %f", i, total_reward)
    return total_reward

if __name__ == "__main__":
    # ... (Previous code remains unchanged)

    num_pendulums = 4  # Change this to the desired number of pendulums
    verbose = False
    nx = 2  # Number of states (angle and angular velocity)
    nu = 1  # Number of control inputs
    TIMESTEPS = 10
    ACTION_LOW = -2
    ACTION_HIGH = 2
    LQR_ITER = 5
    N_BATCH = 2
    u_init = 0.0
    u_lower = torch.tensor(ACTION_LOW, dtype=torch.float32).view(1, -1)
    u_upper = torch.tensor(ACTION_HIGH, dtype=torch.float32).view(1, -1)
    # Create the MPC controller and cost function once outside the parallel processes
    ctrl = mpc.MPC(nx , nu, TIMESTEPS, u_lower, u_upper, lqr_iter=LQR_ITER,
                   exit_unconverged=False, eps=1e-2,
                   n_batch=N_BATCH, backprop=False, verbose=0, u_init=u_init,
                   grad_method=mpc.GradMethods.AUTO_DIFF)

    pendulum_processes = []
    with mp.Pool(processes=num_pendulums) as pool:
        for i in range(num_pendulums):
            # Pass the controller and cost function to the function that runs the pendulum simulation
            process_args = (i, cost, u_init, render, verbose)
            pendulum_processes.append(pool.apply_async(run_single_pendulum, process_args))

        # Wait for all pendulums to finish and get their rewards
        total_rewards = [pendulum_process.get() for pendulum_process in pendulum_processes]
        total_reward = sum(total_rewards)

    logger.info("Total reward from all pendulums: %f", total_reward)

