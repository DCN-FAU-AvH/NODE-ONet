#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from models.utils import logging_setup
from models.space_basis import FNN
from models.neural_odes import SA_NODE_NS
from models.training import Trainer_NS
from models.plot import plot_compare_NS
from models.gen_data_2DNS import gen_2D_NS_data
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # mps is for Mac, cuda is for Linux and Windows

def main():
    # =============================================================================
    # Hyperparameters and Setup
    # =============================================================================
    # Data Generation
    T = 10       # T is the final time
    N_x = 50    #N_x is the number of spatial points
    N_t = 10    # N_t is the number of time points
    time_steps = N_t    # Number of time steps
    N_v_x = 50    # N_v_x is the number of spatial points for the random field
    N_v_num_train = 1000   # N_v_num is the number of random fields
    visc = 0.001    # Constant viscosity
    f_input = False    # Type of the forcing term, True for variable, False for function
    wi_input = True    # Type of the initial condition, True for variable, False for function

    x_train, t_train, f_v_train, w_v_train, w_train = gen_2D_NS_data(N_x, N_t, N_v_x, N_v_num_train, T, visc, f_input, wi_input, device)    # Generate the training data

    # Model
    data_dim = 2            # Dimension of the Problem
    dim_v = N_v_x**2       # Dimension of the R^{dv} space
    dim_u = 200              # Dimension of the R^{du} space
    hidden_dim = 2000         # Dimension of the hidden layers
    hidden_layers = 4       # Number of hidden layers
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
    node = SA_NODE_NS(device, data_dim, dim_v, dim_u, hidden_dim, hidden_layers, 'relu', T, time_steps, f_input).to(device)
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
    trainer = Trainer_NS(node, basis, device, reg=reg_type, logger=logger)
    # first 1000 epochs with Adam
    trainer.train(w_train, w_v_train, f_v_train, x_train, t_train, optimizer_adam, optimizer_type='adam', num_epochs=num_epochs_adam)
    # next 1000 epochs with L-BFGS
    trainer.train(w_train, w_v_train, f_v_train, x_train, t_train, optimizer_lbfgs, optimizer_type='lbfgs', num_epochs=num_epochs_lbfgs)
    
    # Save the model
    torch.save(node.state_dict(), save_folder + filename_node)
    torch.save(basis.state_dict(), save_folder + filename_basis)
    # =============================================================================
    # TESTING
    # =============================================================================
    T = 10       
    N_v_num_test = 200
    N_x = 100
    N_t = 100
    print(f"Testing for T={T}")

    x_test, t_test, f_v_test, w_v_test, w_test = gen_2D_NS_data(N_x, N_t, N_v_x, N_v_num_test, T, visc, f_input, wi_input, device)    # Generate the testing data
    
    wi_test = w_v_test[:, :, 0]
    w_operator = trainer.compute_u_operator(wi_test, f_v_test, t_test, x_test)    # Compute the operator and squeeze the last dimension
    print(f"Error in [0, {T}]:")
    logger.info(f"Error in [0, {T}]:")
    error = trainer.error_compute(w_test, w_operator)    # Compute and print the error
    print(f"Error at t={T}:")
    logger.info(f"Error at t={T}:")
    error_T = trainer.error_compute(w_test[:, :, -1], w_operator[:, :, -1])    # Compute and print the error at T

    # test for prediction
    T = 20
    print(f"Testing for T={T}")

    x_test, t_test, f_v_test, w_v_test, w_test = gen_2D_NS_data(N_x, N_t, N_v_x, N_v_num_test, T, visc, f_input, wi_input, device)    # Generate the testing data
    
    wi_test = w_v_test[:, :, 0]
    w_operator = trainer.compute_u_operator(wi_test, f_v_test, t_test, x_test)    # Compute the operator and squeeze the last dimension
    print(f"Error in [0, {T}]:")
    logger.info(f"Error in [0, {T}]:")
    error = trainer.error_compute(w_test, w_operator)    # Compute and print the error
    print(f"Error at t={T}:")
    logger.info(f"Error at t={T}:")
    error_T = trainer.error_compute(w_test[:, :, -1], w_operator[:, :, -1])    # Compute and print the error at T

    # plot the results for n+1 samples
    n = 10
    for i in range(n+1):
        index_t = (N_t//n)*i
        ti = t_test[index_t].item()
        plot_compare_NS(w_operator[0, :, index_t], w_test[0, :, index_t], save_folder, N_x, ti)
        


if __name__ == "__main__":
    main()

