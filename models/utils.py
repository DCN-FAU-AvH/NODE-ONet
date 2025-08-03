import os, pytz, time
import logging
import torch
import torch.nn as nn
import numpy as np
from scipy import interpolate
from pathos.pools import ProcessPool
from sklearn import gaussian_process as gp

def create_tests_folder(parent_folder="", prefix="", postfix=""):
    # Create tests_folder based on time and change it to Berlin time zone
    time_stamp = int(time.time())
    time_zone = pytz.timezone("Europe/Berlin")
    test_time = pytz.datetime.datetime.fromtimestamp(time_stamp, time_zone)
    test_time = test_time.strftime("%Y%m%d-%H%M%S")
    tests_folder = os.getcwd() + f"/{parent_folder}/{prefix}{test_time}{postfix}"
    os.makedirs(tests_folder)
    print(f"\nWorking in folder {tests_folder}\n")
    return tests_folder

def logging_setup():
    # Create a logger
    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.DEBUG)

    # Create a folder to save the results and 0_scenario.txt
    save_folder = create_tests_folder(parent_folder="results", prefix="")
    file_handler = logging.FileHandler(save_folder + "/logging.txt")
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    
    return logger, save_folder



def activation_func(activation):    # Define activation functions
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU()
    elif activation == 'reluk':
        return ReLUk()
    else:
        raise ValueError("Activation function not recognized")
    
def loss_func(loss_type):    # Define loss functions
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'l1':
        return nn.L1Loss()
    elif loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif loss_type == 'multi-margin':
        return nn.MultiMarginLoss()
    else:
        raise ValueError("Loss function not recognized")


class ReLUk(nn.Module):  # Define ReLU^k activation function
    def __init__(self):
        super(ReLUk, self).__init__()
        self.k = 2
    def forward(self, x):
        return torch.pow(torch.relu(x), self.k)
    

class GRF(object):
    def __init__(self, T, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(0, T, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, n):
        """Generate `n` random feature vectors.
        """
        u = np.random.randn(self.N, n)
        return np.dot(self.L, u).T

    def eval_u_one(self, y, x):
        """Compute the function value at `x` for the feature `y`.
        """
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), y)
        f = interpolate.interp1d(
            np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        )
        return f(x)

    def eval_u(self, ys, sensors):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """
        if self.interp == "linear":
            return np.vstack([np.interp(sensors, np.ravel(self.x), y).T for y in ys])
        p = ProcessPool(nodes=4)
        res = p.map(
            lambda y: interpolate.interp1d(
                np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
            )(sensors).T,
            ys,
        )
        return np.vstack(list(res))
    


