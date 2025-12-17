### Continuous version for Hamiltonian dynamics training of environments in OpenAI Gym classical controls
### The rendering part is the same as OpenAI Gym
import numpy as np
import math
import gymnasium as gym
from gym.utils import seeding
from os import path
import torch

### Generic continuous environment for reduced Hamiltonian dynamics framework
class ContinuousEnv():
    def __init__(self, q_dim=1, u_dim=1):
        self.q_dim = q_dim
        self.u_dim = u_dim
        self.eps = 1e-8
        self.id = np.eye(q_dim)
        self.seed()
        
        # Viewer for rendering image
        self.viewer = None
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    # Dynamics f
    def f(self, q, u):
        return np.zeros((q.shape[0], self.q_dim))
    
    # Partial derivative of dynamics f wrt control u. Assuming linear control
    def f_u(self, q):
        return np.zeros((q.shape[0], self.q_dim, self.u_dim))
    
    # Lagrangian or running cost L
    def L(self, q, u):
        return np.zeros(q.shape[0])
    
    # Terminal cost g
    def g(self, q):
        return np.zeros(q.shape[0])
    
    # Nabla of g
    def nabla_g(self, q):
        ret = np.zeros((q.shape[0], self.q_dim))
        for i in range(self.q_dim):
            ret[:, i] = (self.g(q+self.eps*self.id[i])-self.g(q-self.eps*self.id[i]))/(2*self.eps)
        return ret
    
    # Sampling state q
    def sample_q(self, num_examples, mode='train'):
        return np.zeros((num_examples, self.q_dim))
    
    # Image rendering
    def render(self, q, mode="rgb_array"):
        return
    
    # Close rendering
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


#### Mountain car for PMP ####
class MountainCar(ContinuousEnv):
    def __init__(self, q_dim=2, u_dim=1, goal_velocity=0):
        super().__init__(q_dim, u_dim)
        self.min_action = -1.0
        self.max_action = 1.0
        self.min_position = -1.2
        self.max_position = 0.6
        self.max_speed = 0.07
        self.goal_position = (
            0.45  # was 0.5 in gym, 0.45 in Arnaud de Broissia's version
        )
        self.goal_velocity = goal_velocity
        self.power = 0.0015
        self.screen_width = 600
        self.screen_height = 400
        self.gym_env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
        self.gym_env.reset()

    
    # (q0, q1) = (position, velocity)
    def f(self, q, u):
        force = np.clip(u, self.min_action, self.max_action)
        return np.concatenate((q[:, 1:], 
                force[:, 0:]*self.power - 0.0025 * np.cos(3 * q[:, 0:1])), axis=1)
    
    def f_u(self, q):
        N = q.shape[0]
        return np.concatenate((np.zeros((N, 1, 1)), np.ones((N, 1, 1))*self.power), axis=1)
    
    def L(self, q, u):
        return 0.1*np.sum(u**2) + self.g(q)

    def g(self, q):
        return (self.goal_velocity-q[:, 1])**2 + (self.goal_position-q[:, 0])**2
    
    def sample_q(self, num_examples, mode='train'):
        if mode == 'train': 
            a = 0.5
        else:
            a = 1
        return np.concatenate(
            (a*np.random.uniform(high=self.max_position, low=self.min_position, size=(num_examples, 1)),
            np.random.uniform(high=self.max_speed, low=-self.max_speed, size=(num_examples, 1))),
            axis=1)
    
    def eval(self, q):
        return (self.goal_position-q[:, 0])**2
    
    def _height(self, xs):
        return np.sin(3 * xs) * 0.45 + 0.55
    
    def step(self,
             states,
             actions,             # torch.Tensor shape (N,) or (N,1) or scalar
             dt=1.0,
             ):
        """
        Batched mountain car step using PyTorch tensors.
        Returns: next_states (N,2), rewards (N,), dones (N,) (bool), info (dict of tensors).
        """
        pos = states[:, 0].clone()
        vel = states[:, 1].clone()
        update = torch.from_numpy(self.f(states.detach().numpy(), actions.detach().numpy()))
        vel = vel + update[:, 1] * dt
        vel = torch.clamp(vel, -self.max_speed, self.max_speed)
        pos = pos + vel * dt
        next_states = torch.stack([pos, vel], dim=-1)
        return next_states

    def render(self, u, mode="rgb_array"):
        # print(u)gym_env
        # print("Before: ", self.gym_env.unwrapped.state)
        self.gym_env.step(u)
        frame = self.gym_env.render()
        return frame


#### CartPole for PMP ####
class CartPole(ContinuousEnv):
    def __init__(self, q_dim=4, u_dim=1):
        super().__init__(q_dim, u_dim)
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
    
        # For continous version
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.screen_width = 600
        self.screen_height = 400 
        self.gym_env = gym.make("CartPole-v1", render_mode="rgb_array")
        self.gym_env.reset()
        
    def f(self, q, u):
        _, x_dot, theta, theta_dot = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        force = self.force_mag * u.reshape(-1)
        costheta, sintheta = np.cos(theta), np.sin(theta)
        temp = (
            force + self.polemass_length * (theta_dot ** 2) * sintheta
        ) / self.total_mass
        
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        thetaacc, xacc = thetaacc.reshape(-1, 1), xacc.reshape(-1, 1), 
    
        return np.concatenate((x_dot.reshape(-1, 1), xacc, theta_dot.reshape(-1, 1), thetaacc), axis=1)
    
    def f_u(self, q):
        theta = q[:, 2]
        N = q.shape[0]
        costheta = np.cos(theta)
        tmp_u = self.force_mag /self.total_mass
        xacc_u = tmp_u * np.ones((N, 1))
        thetaacc_u = -costheta*tmp_u/(
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        thetaacc_u = thetaacc_u.reshape(-1, 1)
        return np.concatenate((np.zeros((N, 1)), xacc_u, np.zeros((N, 1)), thetaacc_u), axis=1)\
            .reshape(-1, self.q_dim, self.u_dim)
        
        
    def L(self, q, u):
        return 0.5*np.sum(u**2)
    
    def g(self, q):
        #noise = np.random.normal(scale=0.001, size=(q.shape[0]))
        #t = [self.x_threshold/2, self.theta_threshold_radians/2]
        #a = 0.005
        return (q[:, 2]/self.theta_threshold_radians)**2 #(a**2-q[:, 0]**2)
    
    def eval(self, q):
        return (q[:, 2]/self.theta_threshold_radians)**2 
    
    def sample_q(self, num_examples, mode='train'):
        if mode == 'train':
            a = 0.01
        else:
            a = 0.05
        return np.random.uniform(low=-a, high=a, size=(num_examples, 4))
    
    def render(self, u, mode="rgb_array"):
        if u[0] >= 0:
            self.gym_env.step(1)
        else:
            self.gym_env.step(0)
            
        frame = self.gym_env.render()
        return frame
    
def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


#### Pendulum for PMP ####
class Pendulum(ContinuousEnv):
    def __init__(self, q_dim=2, u_dim=1, gravity=9.8):
        super().__init__(q_dim, u_dim)
        self.max_speed = 8
        self.max_torque = 2.0
        self.gravity = gravity
        self.m = 1.0
        self.l = 1.0

        self.screen_width = 500
        self.screen_height = 500
    
    # (q0, q1) = (position, velocity)
    def f(self, q, u):
        u = np.clip(u, -self.max_torque, self.max_torque)
        gravity, m, l = self.gravity, self.m, self.l
        return np.concatenate((q[:, 1:], 3*gravity/(2*l)*np.sin(q[:, 0:1]) + 3/(m*(l**2))*u[:, 0:]), axis=1)
        
    def f_u(self, q):
        m, l = self.m, self.l
        N = q.shape[0]
        return np.concatenate((np.zeros((N, 1, 1)), 
                               3/(m*(l**2))*np.ones((N, 1, 1))), axis=1)
    
    def L(self, q, u):
        u = np.clip(u, -self.max_torque, self.max_torque)
        return u[:, 0]**2 + self.g(q)
    
    def g(self, q):
        return (angle_normalize(q[:, 0])+np.pi/2)**2
    
    def eval(self, q):
        return (angle_normalize(q[:, 0])+np.pi/2)**2
    
    def sample_q(self, num_examples, mode='train'):
        if mode=='train':
            a = 0.1
        else:
            a = 0.01
        return a*np.concatenate(
                (np.random.uniform(high=np.pi, low=-np.pi, size=(num_examples, 1)),
                np.random.uniform(high=1, low=-1, size=(num_examples, 1))),
                axis=1)
        
    def render(self, q, mode="rgb_array"):
        screen_width = self.screen_width
        screen_height = self.screen_height

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, 0.2)
            rod.set_color(0.8, 0.3, 0.3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(0.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "clockwise.png")
            self.img = rendering.Image(fname, 1.0, 1.0)
            #self.imgtrans = rendering.Transform()
            #self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(q[0]+np.pi/2)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
