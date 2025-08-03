import torch.nn as nn
import torch
from torchdiffeq import odeint
from models.utils import activation_func



class Vanilla_NODE(nn.Module):
    def __init__(self, 
                 device,        # Device
                 data_dim,      # Dimension of the problem
                 dim_v,         # Dimension of the R^{dv} space
                 dim_u,         # Dimension of the R^{du} space
                 hidden_dim,    # Dimension of the hidden layers
                 hidden_layers, # Number of hidden layers
                 D,             # Control variable D
                 non_linearity, # activation function type
                 T,             # Final time
                 time_steps,    # Number of time steps
                 D_input=True,  # Include D in the input, True or False
                 v_input=True): # Include v in the input, True or False
        super(Vanilla_NODE, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.dim_v = dim_v
        self.dim_u = dim_u
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.D = D
        self.non_linearity = activation_func(non_linearity)
        self.T = T
        self.time_steps = time_steps
        self.dt = T/(time_steps-1)
        self.D_input = D_input
        self.v_input = v_input

        # Define the projection layers
        if D_input:
            self.P_D = nn.Linear(dim_v, hidden_dim, bias=False).to(device)  # R^{dv} -> R^{du}
        if v_input:
            self.P_v = nn.Linear(dim_v, dim_u, bias=False).to(device)  # R^{dv} -> R^{du}
        self.P_u = nn.Linear(1, dim_u, bias=False).to(device)      # R^{1} -> R^{du}

        # Define the neural ODE
        ##-- R^{d_u} -> R^{d_hid} layer -- 
        blocks_NODE_1 = [nn.Linear(dim_u, hidden_dim) 
                    for _ in range(self.time_steps)]
        self.fc1_NODE = nn.Sequential(*blocks_NODE_1).to(device)
        ##-- R^{d_hid} -> R^{d_u} layer --
        blocks_NODE_2 = [nn.Linear(hidden_dim, dim_u, bias=False) 
                    for _ in range(self.time_steps)]
        self.fc2_NODE = nn.Sequential(*blocks_NODE_2).to(device)

        # Initialize Xavier weights and zero biases
        nn.init.xavier_normal_(self.P_v.weight)
        for block in self.fc1_NODE:
            nn.init.xavier_normal_(block.weight)
            nn.init.zeros_(block.bias)
        for block in self.fc2_NODE:
            nn.init.xavier_normal_(block.weight)
        
    
    def forward(self, t, phi):
        k = int(t/self.dt)  # Get the time step index

        # NODE part
        if self.D_input:
            D_regular = self.P_D(self.D)     # R^{d_v} -> R^{d_u}
        if self.v_input:
            v_regular = self.P_v(self.v)     # R^{d_v} -> R^{d_u}

        w1_t = self.fc1_NODE[k].weight
        b_t = self.fc1_NODE[k].bias
        w2_t = self.fc2_NODE[k].weight

        if self.D_input and self.v_input:
            rhs_NODE = self.non_linearity(phi.matmul(w1_t.t()) + b_t).matmul((w2_t*D_regular).t()) + v_regular
        elif self.D_input==False and self.v_input:
            rhs_NODE = self.non_linearity(phi.matmul(w1_t.t()) + b_t).matmul(w2_t.t()) + v_regular
        elif self.D_input and self.v_input==False:
            rhs_NODE = self.non_linearity(phi.matmul(w1_t.t()) + b_t).matmul((w2_t*D_regular).t())

        return rhs_NODE
    
    def solve(self, u0, t, v):
        U0 = self.P_u(u0)   # project the initial condition to R^{du}
        self.v = v
        
        out_NODE = odeint(self,                             # NODE model
                          U0,                               # Initial condition
                          t,                                # Time domain
                          method='euler',                   # ODE solver
                          options={'step_size': self.dt})   # Time step size
        
        return out_NODE

class SA_NODE(nn.Module):
    def __init__(self, 
                 device,        # Device
                 data_dim,      # Dimension of the problem
                 dim_v,         # Dimension of the R^{dv} space
                 dim_u,         # Dimension of the R^{du} space
                 hidden_dim,    # Dimension of the hidden layers
                 hidden_layers, # Number of hidden layers
                 non_linearity, # activation function type
                 T,             # Final time
                 time_steps,    # Number of time steps
                 D_input=True,  # Include D in the input, True or False
                 v_input=True): # Include v in the input, True or False   
        super(SA_NODE, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.dim_v = dim_v
        self.dim_u = dim_u
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.non_linearity = activation_func(non_linearity)
        self.T = T
        self.time_steps = time_steps
        self.dt = T/(time_steps-1)
        self.D_input = D_input
        self.v_input = v_input

        # Define the projection layers
        if D_input:
            self.P_D = nn.Linear(dim_v, dim_u, bias=False).to(device)  # R^{dv} -> R^{du}
        if v_input:
            self.P_v = nn.Linear(dim_v, dim_u, bias=False).to(device)  # R^{dv} -> R^{du}
        self.P_u = nn.Linear(dim_v, dim_u, bias=False).to(device)      # R^{1} -> R^{du}


        # Define the neural ODE
        ##-- R^{d_aug} -> R^{d_hid} layer -- 
        self.fc1_time = nn.Linear(dim_u, hidden_dim, bias=False) 
        self.b_time = nn.Linear(1, hidden_dim)
        ##-- R^{d_hid} -> R^{d_aug} layer --
        self.fc2_time = nn.Linear(hidden_dim, dim_u, bias=False)
        # Initialize Xavier weights and zero biases
        nn.init.xavier_normal_(self.fc1_time.weight)
        nn.init.xavier_normal_(self.fc2_time.weight)
        nn.init.xavier_normal_(self.b_time.weight)
        nn.init.zeros_(self.b_time.weight)
        
    
    def forward(self, t, phi):
        t = torch.tensor([t]).float().to(self.device)
        # regularize the control variables
        if self.D_input:
            D_regular = self.P_D(self.D)     # R^{d_v} -> R^{d_u}
        if self.v_input:
            v_regular = self.P_v(self.v)     # R^{d_v} -> R^{d_u}

        # NODE part
        w1_t = self.fc1_time.weight
        b_t = self.b_time(t)
        w2_t = self.fc2_time.weight

        # W = self.P_D.weight 
        # W_scaled = W.unsqueeze(0) * self.D.unsqueeze(1)
        # M = torch.matmul(W_scaled, W.t()) 
        # out = torch.einsum('bij, bi -> bj', M, phi)    

        # if self.D_input and self.v_input:
        #     rhs_NODE = out + self.non_linearity(phi.matmul(w1_t.t()) + b_t).matmul(w2_t.t()) + v_regular
        # elif self.D_input==False and self.v_input:
        #     rhs_NODE = self.non_linearity(phi.matmul(w1_t.t()) + b_t).matmul(w2_t.t()) + v_regular
        #     # rhs_NODE = self.non_linearity(phi.matmul(w1_t.t()) + b_t).matmul(w2_t.t()) + v_regular
        # elif self.D_input and self.v_input==False:
        #     rhs_NODE = out + self.non_linearity(phi.matmul(w1_t.t()) + b_t).matmul(w2_t.t())

        # original version
        if self.D_input and self.v_input:
            rhs_NODE = self.non_linearity((phi*D_regular).matmul(w1_t.t()) + b_t).matmul(w2_t.t()) + v_regular
        elif self.D_input==False and self.v_input:
            rhs_NODE = self.non_linearity(phi.matmul(w1_t.t()) + b_t).matmul(w2_t.t()) + v_regular
        elif self.D_input and self.v_input==False:
            rhs_NODE = self.non_linearity((phi*D_regular).matmul(w1_t.t()) + b_t).matmul(w2_t.t())

        return rhs_NODE
    
    def solve(self, u0, t, v, D):
        U0 = self.P_u(u0)   # project the initial condition to R^{du}
        self.v = v
        self.D = D

        out_NODE = odeint(self,                             # NODE model
                          U0,                               # Initial condition
                          t,                                # Time domain
                          method='euler',                   # ODE solver
                          options={'step_size': self.dt})   # Time step size
        
        return out_NODE


class SA_NODE_NS(nn.Module):
    def __init__(self, 
                 device,        # Device
                 data_dim,      # Dimension of the problem
                 dim_v,         # Dimension of the R^{dv} space
                 dim_u,         # Dimension of the R^{du} space
                 hidden_dim,    # Dimension of the hidden layers
                 hidden_layers, # Number of hidden layers
                 non_linearity, # activation function type
                 T,             # Final time
                 time_steps,    # Number of time steps): 
                 f_input = True): # Include the forcing term, True or False
                 
        super(SA_NODE_NS, self).__init__()
        self.device = device
        self.data_dim = data_dim
        self.dim_v = dim_v
        self.dim_u = dim_u
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.non_linearity = activation_func(non_linearity)
        self.T = T
        self.time_steps = time_steps
        self.dt = T/time_steps
        self.f_input = f_input

        # Define the projection layers
        if f_input:
            self.P_f = nn.Linear(dim_v, dim_u, bias=False).to(device)
        self.P_wi = nn.Linear(dim_v, dim_u, bias=False).to(device)

        # Define the neural ODE
        ##-- R^{d_aug} -> R^{d_hid} layer -- 
        self.fc1_time = nn.Linear(dim_u, hidden_dim, bias=False) 
        self.b_time = nn.Linear(1, hidden_dim)
        self.P_V = nn.Linear(dim_u, hidden_dim, bias=False)
        self.P_W = nn.Linear(dim_u, hidden_dim, bias=False)
        ##-- R^{d_hid} -> R^{d_aug} layer --
        self.fc2_time = nn.Linear(hidden_dim, dim_u, bias=False)
        # Initialize Xavier weights and zero biases
        nn.init.xavier_normal_(self.fc1_time.weight)
        nn.init.xavier_normal_(self.fc2_time.weight)
        nn.init.xavier_normal_(self.b_time.weight)
        # nn.init.xavier_normal_(self.P_V.weight)
        # nn.init.xavier_normal_(self.P_W.weight)
        nn.init.zeros_(self.P_V.weight)
        nn.init.zeros_(self.P_W.weight)
        nn.init.zeros_(self.b_time.weight)
        
    
    def forward(self, t, phi):
        t = torch.tensor([t]).float().to(self.device)
        # regularize the control variables

        # NODE part
        w1_t = self.fc1_time.weight
        b_t = self.b_time(t)
        w2_t = self.fc2_time.weight
        P_V = self.P_V.weight
        P_W = self.P_W.weight

        # Regularize forcing term
        if self.f_input:
            f_part = self.P_f(self.f)
        else:
            f_part = 0

        rhs_NODE = self.non_linearity(phi.matmul(w1_t.t()) + (phi.matmul(P_V.t()))*(phi.matmul(P_W.t())) + b_t).matmul(w2_t.t()) + f_part

        return rhs_NODE
    
    def solve(self, t, w0, f):
        U0 = self.P_wi(w0)   # project the initial condition to R^{du}
        self.f = f

        out_NODE = odeint(self,                             # NODE model
                          U0,                               # Initial condition
                          t,                                # Time domain
                          method='euler',                   # ODE solver
                          options={'step_size': self.dt})   # Time step size
        
        return out_NODE























