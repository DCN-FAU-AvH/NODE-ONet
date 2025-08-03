import torch
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
from models.utils import loss_func
import time

console = Console(width=100)

class Trainer():
    def __init__(self, 
                 node,              # Neural ODE model
                 basis,             # Basis function model
                 device,            # Device
                 reg='l1',          # Regularization type
                 loss_type='mse',   # Loss function type
                 logger=None):      # Logger
        self.node = node
        self.basis = basis
        self.device = device
        self.reg = reg
        self.logger = logger
        self.loss_func = loss_func(loss_type)

    def train(self, ux_real, ux_v, v, D, x, t, optimizer, optimizer_type, num_epochs):
        self.num_epochs = num_epochs
        self.start_time = time.time()
        with Progress(      # Define the progress bar
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            expand=True
            ) as progress:
            
            if optimizer_type == 'lbfgs':
                print("Optimizer: L-BFGS")
                for epoch in progress.track(range(num_epochs)):
                    loss = self._train_epoch(ux_real, ux_v, v, D, x, t, optimizer, optimizer_type)
                console.print("Optimizer: L-BFGS | Loss: {:.3e} | Time: {:.1f} min"\
                            .format(loss, (time.time()-self.start_time)/60.))
                if self.logger is not None:
                    self.logger.info("Optimizer: L-BFGS | Loss: {:.3e} | Time: {:.1f} min"\
                                    .format(loss, (time.time()-self.start_time)/60.))
            else:
                print("Optimizer: Adam")
                for epoch in progress.track(range(num_epochs)):
                    loss = self._train_epoch(ux_real, ux_v, v, D, x, t, optimizer, optimizer_type)
                    # Print the loss every 100 epochs
                    if (epoch+1)%100==0:
                        console.print("Epoch {}/{} | Loss: {:.3e} | Time: {:.1f} min"\
                            .format(epoch+1, self.num_epochs, loss, (time.time()-self.start_time)/60.))
                        if self.logger is not None:
                            self.logger.info("Epoch {}/{} | Loss: {:.3e} | Time: {:.1f} min"\
                                .format(epoch+1, self.num_epochs, loss, (time.time()-self.start_time)/60.))
        
        # print the final time in secs
        console.print("Total time: {:.1f} sec".format(time.time()-self.start_time))
        if self.logger is not None:
            self.logger.info("Total time: {:.1f} sec".format(time.time()-self.start_time))

    
    def _train_epoch(self, ux_real, ux_v, v, D, x, t, optimizer, optimizer_type):
        
        # Backward and optimize
        if optimizer_type == 'adam':
            loss = self.losses(ux_real, ux_v, v, D, x, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
        
        elif optimizer_type == 'lbfgs':
            def closure():
                optimizer.zero_grad()
                loss = self.losses(ux_real, ux_v, v, D, x, t)
                loss.backward()
                return loss
            loss = optimizer.step(closure)

        return loss.item()      
        
    def losses(self, ux_real, ux_v, v, D, x, t):
        u_operator = self.compute_u_operator(ux_v, v, D, t, x)
            
        # Compute the loss
        '''
        # regularization term
        loss_reg = 0.0
        
        if self.reg == 'l1':
            for param in self.node.parameters():
                loss_reg += param.abs().sum()
            for param in self.basis.parameters():
                loss_reg += param.abs().sum()
        lambda_reg = 1e-6
        '''
        # loss of u_operator
        loss_u = self.loss_func(u_operator, ux_real)
        lambda_u = 100.0

        loss = lambda_u * loss_u  # + lambda_reg * loss_reg

        return loss

    def compute_u_operator(self, ux_v, v, D, t, x):
        u0 = ux_v[:, :, 0] # [21, 50]
        # Compute u_operator
        out_NODE = self.node.solve(u0, t, v, D) # [21, 20, 50]
        u_operator = torch.zeros((v.shape[0], x.shape[0], t.shape[0])).to(self.device)
        # if self.basis is FNN, then the basis is the same for all the nodes, if FNN_Time, then the basis is different
        if self.basis.__class__.__name__ == 'FNN':
            out_basis = self.basis(x) # [21, 50]
            u_operator = torch.einsum('nmd, jd->mjn', out_NODE, out_basis)

        elif self.basis.__class__.__name__ == 'FNN_Time':
            for i in range(out_NODE.shape[0]):
                out_basis = self.basis(i, x)
            all_out_basis = torch.stack([self.basis(i, x) for i in range(out_NODE.shape[0])], dim=0)
            u_operator = torch.einsum('imd,ibd->imb', out_NODE, all_out_basis).permute(1, 2, 0)  
        
        return u_operator
    
    def error_compute(self, u_real, u_operator):
        # Compute the L2 error
        error_L2 = ((u_real - u_operator)**2).mean().sqrt()
        # Compute the relative L2 error
        error_L2_rel = error_L2 / ((u_real**2).mean().sqrt())

        console.print("Final L2 Error: {:.3e}".format(error_L2))
        console.print("Final Relative L2 Error: {:.3e}".format(error_L2_rel))
        if self.logger is not None:
            self.logger.info("Final L2 Error: {:.3e}".format(error_L2))
            self.logger.info("Final Relative L2 Error: {:.3e}".format(error_L2_rel))
        
        return error_L2, error_L2_rel




class Trainer_NS():
    def __init__(self, 
                 node,              # Neural ODE model
                 basis,             # Basis function model
                 device,            # Device
                 reg='l1',          # Regularization type
                 loss_type='mse',   # Loss function type
                 logger=None):      # Logger
        self.node = node
        self.basis = basis
        self.device = device
        self.reg = reg
        self.logger = logger
        self.loss_func = loss_func(loss_type)

    def train(self, w, wv, fv, x, t, optimizer, optimizer_type, num_epochs):
        self.num_epochs = num_epochs
        self.start_time = time.time()
        with Progress(      # Define the progress bar
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(bar_width=None),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            expand=True
            ) as progress:
            
            if optimizer_type == 'lbfgs':
                print("Optimizer: L-BFGS")
                for epoch in progress.track(range(num_epochs)):
                    loss = self._train_epoch(w, wv, fv, x, t, optimizer, optimizer_type)
                console.print("Optimizer: L-BFGS | Loss: {:.3e} | Time: {:.1f} min"\
                            .format(loss, (time.time()-self.start_time)/60.))
                if self.logger is not None:
                    self.logger.info("Optimizer: L-BFGS | Loss: {:.3e} | Time: {:.1f} min"\
                                    .format(loss, (time.time()-self.start_time)/60.))
            else:
                print("Optimizer: Adam")
                for epoch in progress.track(range(num_epochs)):
                    loss = self._train_epoch(w, wv, fv, x, t, optimizer, optimizer_type)
                    # Print the loss every 100 epochs
                    if (epoch+1)%100==0:
                        console.print("Epoch {}/{} | Loss: {:.3e} | Time: {:.1f} min"\
                            .format(epoch+1, self.num_epochs, loss, (time.time()-self.start_time)/60.))
                        if self.logger is not None:
                            self.logger.info("Epoch {}/{} | Loss: {:.3e} | Time: {:.1f} min"\
                                .format(epoch+1, self.num_epochs, loss, (time.time()-self.start_time)/60.))
        
        # print the final time in secs
        console.print("Total time: {:.1f} sec".format(time.time()-self.start_time))
        if self.logger is not None:
            self.logger.info("Total time: {:.1f} sec".format(time.time()-self.start_time))

    
    def _train_epoch(self, w, wv, fv, x, t, optimizer, optimizer_type):
        
        # Backward and optimize
        if optimizer_type == 'adam':
            loss = self.losses(w, wv, fv, x, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
        
        elif optimizer_type == 'lbfgs':
            def closure():
                optimizer.zero_grad()
                loss = self.losses(w, wv, fv, x, t)
                loss.backward()
                return loss
            loss = optimizer.step(closure)

        return loss.item()      
        
    def losses(self, w_real, wv_real, fv, x, t):
        wi = wv_real[:, :, 0] 
        w_operator = self.compute_u_operator(wi, fv, t, x)
            
        # Compute the loss
        # regularization term
        
        
        loss_reg = 0.0
        if self.reg == 'l1':
            for param in self.node.parameters():
                loss_reg += param.abs().sum()
            for param in self.basis.parameters():
                loss_reg += param.abs().sum()
        lambda_reg = 1e-5
        
        # loss of u_operator
        loss_u = self.loss_func(w_operator, w_real)
        lambda_u = 100

        loss = lambda_u * loss_u  + lambda_reg * loss_reg

        return loss

    def compute_u_operator(self, wi, f, t, x):
        # Compute u_operator
        out_NODE = self.node.solve(t, wi, f) 
        u_operator = torch.zeros((wi.shape[0], x.shape[0], t.shape[0])).to(self.device)
        # if self.basis is FNN, then the basis is the same for all the nodes, if FNN_Time, then the basis is different
        if self.basis.__class__.__name__ == 'FNN' or self.basis.__class__.__name__ == 'CNN':
            out_basis = self.basis(x) 
            u_operator = torch.einsum('nmd, jd->mjn', out_NODE, out_basis)

        elif self.basis.__class__.__name__ == 'FNN_Time':
            for i in range(out_NODE.shape[0]):
                out_basis = self.basis(i, x)
            all_out_basis = torch.stack([self.basis(i, x) for i in range(out_NODE.shape[0])], dim=0)
            u_operator = torch.einsum('imd,ibd->imb', out_NODE, all_out_basis).permute(1, 2, 0)  
        
        return u_operator
    
    def error_compute(self, u_real, u_operator):
        # Compute the L2 error
        error_L2 = ((u_real - u_operator)**2).mean().sqrt()
        # Compute the relative L2 error
        error_L2_rel = error_L2 / ((u_real**2).mean().sqrt())

        console.print("Final L2 Error: {:.3e}".format(error_L2))
        console.print("Final Relative L2 Error: {:.3e}".format(error_L2_rel))
        if self.logger is not None:
            self.logger.info("Final L2 Error: {:.3e}".format(error_L2))
            self.logger.info("Final Relative L2 Error: {:.3e}".format(error_L2_rel))
        
        return error_L2, error_L2_rel


        
        
        


        
        

