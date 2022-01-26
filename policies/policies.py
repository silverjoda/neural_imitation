import torch.nn as nn
import torch.nn.functional as F
import torch as T
from torch.nn.utils import weight_norm
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

class MLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=128):
        super(MLP, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim

        self.fc1 = T.nn.Linear(self.obs_dim, self.hid_dim, bias=True)
        self.fc2 = T.nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.fc3 = T.nn.Linear(self.hid_dim, self.act_dim, bias=True)

    def forward(self, x):
        fc1 = T.tanh(self.fc1(x))
        fc2 = T.tanh(self.fc2(fc1)) * 1
        out = self.fc3(fc2)
        return out

class CVX(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=128):
        super(CVX, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim

        self.fc_in = T.nn.Linear(self.obs_dim, self.hid_dim, bias=True)
        self.fc_out = T.nn.Linear(self.hid_dim, self.act_dim, bias=True)

        _x = cp.Parameter(self.hid_dim)
        _y = cp.Variable(self.hid_dim)
        obj = cp.Minimize(-_x.T @ _y - cp.sum(cp.entr(_y)))
        prob = cp.Problem(obj)
        self.cvx_layer = CvxpyLayer(prob, parameters=[_x], variables=[_y])

    def forward(self, x):
        x_in = T.tanh(self.fc_in(x))
        cvx_out, = self.cvx_layer(x_in)
        out = self.fc_out(cvx_out)
        return out
