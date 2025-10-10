#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from models.gen_data import DR_System
from models.utils import logging_setup, GRF
from models.space_basis import FNN
from models.neural_odes import SA_NODE, Vanilla_NODE
from models.training import Trainer
from models.plot import plot_compare
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # mps is for Mac, cuda is for Linux GPU

def main():
    # =============================================================================
    # Hyperparameters and Setup
    # =============================================================================
    # Data Generation
    X = 1       # X is the length of the domain [0, X]
    T = 1       # T is the final time
    N_x = 100    # N_x is the number of spatial points
    N_t = 10    # N_t is the number of time points
    k = 0.01    # Reaction coefficient
    D_const = 0.01    # Diffusion coefficient constant
    time_steps = N_t     # Number of time steps
    l = 0.5     # l is the length scale of the random field
    N_v_x = 100    # N_v_x is the number of spatial points for the random field
    N_v_num = 500   # N_v_num is the number of random fields
    v_type = 'nonzero'    # Type of the random field, 'zero', 'nonzero' or 'const'
    D_type = 'const'    # Type of diffusion coefficient, 'const' or 'nonconst'
    seed = 42
    np.random.seed(seed)  # seed global RNG for reproducibility of data splitting
    space = GRF(X, length_scale=l, N=1000, interp="cubic", seed=seed)
    ux_train, ux_v_train, v_train, D_train, t_train, x_train = DR_System(N_v_x, N_x, k, D_const, T, N_t, N_v_num, space, v_type, D_type, device).solve()


    # Model
    data_dim = 1            # Dimension of the Problem
    dim_v = N_v_x           # Dimension of the R^{dv} space
    dim_u = 50              # Dimension of the R^{du} space
    hidden_dim = 100        # Dimension of the hidden layers
    hidden_layers = 2       # Number of hidden layers
    lr_adam = 1e-3          # Learning rate for Adam
    num_epochs_adam = 500000       # Number of epochs for Adam
    num_epochs_lbfgs = 100        # Number of epochs for L-BFGS
    lr_lbfgs = 1e-2            # Learning rate for L-BFGS
    reg_type = 'l1'         # Regularization type, 'l1' or 'barron'
    # Logging and Saving
    logger, save_folder = logging_setup()    # Setup the logger and save folder
    filename_node = '/node.pth'    # Filename to save the NODE model
    filename_basis = '/basis.pth'  # Filename to save the basis model
    # =============================================================================
    # MODEL AND OPTIMIZER
    # =============================================================================
    # Define neural ODE
    node = SA_NODE(device, data_dim, dim_v, dim_u, hidden_dim, hidden_layers, 'relu', T, time_steps,
                   D_input=False, v_input=True).to(device)
    # Define neural network
    basis = FNN(device, data_dim, dim_u, hidden_dim, hidden_layers, 'relu').to(device)
    # Define optimizers, the optimizers work for both models
    optimizer_adam = torch.optim.Adam(list(node.parameters()) + list(basis.parameters()),       # Combine parameters
                                      lr=lr_adam)
    optimizer_lbfgs = torch.optim.LBFGS(list(node.parameters()) + list(basis.parameters()),     # Combine parameters
                                        lr = lr_lbfgs,
                                        max_iter = 20,
                                        max_eval = None,
                                        tolerance_grad = 1e-07,
                                        tolerance_change = 1e-09,
                                        history_size = 100,
                                        line_search_fn = 'strong_wolfe')
    # =============================================================================
    # TRAINING
    # =============================================================================
    trainer = Trainer(node, basis, device, reg=reg_type, logger=logger)
    # first num_epochs_adam epochs with Adam
    trainer.train(ux_train, ux_v_train, v_train, D_train, x_train, t_train, optimizer_adam, optimizer_type='adam', num_epochs=num_epochs_adam)
    # next num_epochs_lbfgs epochs with L-BFGS
    # trainer.train(ux_train, ux_v_train, v_train, D_train, x_train, t_train, optimizer_lbfgs, optimizer_type='lbfgs', num_epochs=num_epochs_lbfgs)
    
    # Save the model
    torch.save(node.state_dict(), save_folder + filename_node)
    torch.save(basis.state_dict(), save_folder + filename_basis)
    # =============================================================================
    # TESTING
    # =============================================================================
    print("Source-to-Solution Testing...")
    # Test by defining new parameters: N_x, N_t, l, N_v_num   
    N_t = 100
    N_x = 100
    T = 1
    N_v_num = 10000
    ux_test, ux_v_test, v_test, D_test, t_test, x_test = DR_System(N_v_x, N_x, k, D_const, T, N_t, N_v_num, space, v_type, D_type, device).solve()

    u_operator = trainer.compute_u_operator(ux_v_test, v_test, D_test, t_test, x_test)  # Compute the operator and squeeze the last dimension
    error = trainer.error_compute(ux_test, u_operator)    # Compute and print the error
    plot_compare(u_operator[0], ux_test[0], save_folder, option='test')    # select the first sample to plot

    

if __name__ == "__main__":
    main()

