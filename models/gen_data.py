import numpy as np
import torch

class DR_System():
    def __init__(self, N_v_x, N_x, k, D_const, T, Nt, N_v_num, space, v_type, D_type, device):
        self.N_v_x = N_v_x      # m
        self.N_x = N_x
        self.k = k
        self.D_const = D_const
        self.T = T
        self.Nt = Nt
        self.N_v_num = N_v_num  # num_train
        self.space = space
        self.v_type = v_type
        self.D_type = D_type
        self.device = device

    def eval_s(self, m, k, T, Nt, sensor_values1, sensor_values2):
        return solve_ADR(
            0,
            1,
            0,
            T,
            lambda x: self.D_const * (1 + abs(sensor_values1)) if self.D_type == "nonconst" else self.D_const*np.ones_like(x),
            lambda x: np.zeros_like(x),
            lambda u: k * u**2,
            lambda u: 2 * k * u,
            lambda x, t: np.tile(sensor_values2[:, None], (1, len(t))) if self.v_type == "nonzero" else np.tile(np.ones_like(x), (1, len(t))),
            lambda x: np.zeros_like(x),
            m,
            Nt,
        )[2]

    def solve(self):
        """Diffusion-reaction on the domain [0, 1] x [0, T].

        Args:
            T: Time [0, T]
            Nt: Nt in FDM
            npoints_output: For a input function, randomly choose these points from the solver output as data
        """
        print("Generating operator data...", flush=True)


        # compute ux_v, v, D
        sensors_vx = np.linspace(0, 1, num=self.N_v_x)[:, None]
        features1 = self.space.random(self.N_v_num)
        sensor_values1 = self.space.eval_u(features1, sensors_vx)
        features2 = self.space.random(self.N_v_num)
        sensor_values2 = self.space.eval_u(features2, sensors_vx)
        ux_v = list(
            map(
                self.eval_s,
                np.hstack(np.tile(self.N_v_x, (self.N_v_num, 1))),
                np.hstack(np.tile(self.k, (self.N_v_num, 1))),
                np.hstack(np.tile(self.T, (self.N_v_num, 1))),
                np.hstack(np.tile(100, (self.N_v_num, 1))), # np.hstack(np.tile(self.Nt, (self.N_v_num, 1))),
                sensor_values1,
                sensor_values2,
            )
        )
        v = sensor_values2 if self.v_type == "nonzero" else np.tile(np.ones_like(sensors_vx), (self.N_v_num, 1))
        # ux_v = np.reshape(ux_v, (self.N_v_num, self.N_v_x, self.Nt))
        ux_v = np.reshape(ux_v, (self.N_v_num, self.N_v_x, 100))
        ux_v = ux_v[:, :, np.linspace(0, 99, num=self.Nt, dtype=int)]

        D = self.D_const * (1 + abs(sensor_values1)) if self.D_type == "nonconst" else np.tile(self.D_const, (self.N_v_num, self.N_v_x))

        # compute ux
        sensors_x = np.linspace(0, 1, num=self.N_x)[:, None]
        sensor_values1 = self.space.eval_u(features1, sensors_x)
        sensor_values2 = self.space.eval_u(features2, sensors_x)
        ux = list(
            map(
                self.eval_s,
                np.hstack(np.tile(self.N_x, (self.N_v_num, 1))),
                np.hstack(np.tile(self.k, (self.N_v_num, 1))),
                np.hstack(np.tile(self.T, (self.N_v_num, 1))),
                np.hstack(np.tile(100, (self.N_v_num, 1))), # np.hstack(np.tile(self.Nt, (self.N_v_num, 1))),
                sensor_values1,
                sensor_values2,
            )
        )
        # ux = np.reshape(ux, (self.N_v_num, self.N_x, self.Nt))
        ux = np.reshape(ux, (self.N_v_num, self.N_x, 100))
        ux = ux[:, :, np.linspace(0, 99, num=self.Nt, dtype=int)]

        x = np.linspace(0, 1, self.N_x)
        t = np.linspace(0, self.T, self.Nt)

        # convert to torch tensors
        ux = torch.tensor(ux, dtype=torch.float32, device=self.device)
        ux_v = torch.tensor(ux_v, dtype=torch.float32, device=self.device)
        v = torch.tensor(v, dtype=torch.float32, device=self.device)
        D = torch.tensor(D, dtype=torch.float32, device=self.device)
        t = torch.tensor(t, dtype=torch.float32, device=self.device)
        x = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        return ux, ux_v, v, D, t, x

        
def solve_ADR(xmin, xmax, tmin, tmax, k, v, g, dg, f, u0, Nx, Nt):
    """Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x, t)
    with zero boundary condition.
    """

    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h**2

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    k = k(x)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v = v(x)
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond
    f = f(x[:, None], t)

    u = np.zeros((Nx, Nt))
    u[:, 0] = u0(x)
    for i in range(Nt - 1):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1, i] + 0.5 * f[1:-1, i + 1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u[1:-1, i + 1] = np.linalg.solve(A, b1 + b2)
    return x, t, u


'''
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist

class Diffusion_Reaction_1D():
    """
    A class to model and solve a 1D diffusion-reaction equation.
    Attributes:
        N_x (int): Number of spatial discretization points.
        X (float): Length of the spatial domain [0, X].
        N_t (int): Number of temporal discretization points.
        T (float): Total time for the simulation.
        N_f (int): Number of points for the control grid.
        l (float or list): Length scale(s) for the Gaussian random field.
        x (ndarray): Spatial grid points.
        t (ndarray): Temporal grid points.
        x_f (ndarray): Control grid points.
    Functions:
        control_grf(l):
            Generates the control variable v based on Gaussian random field with length scale l.
        heat_pde(t, u):
            Defines the heat PDE with reaction term and Gaussian random field.
        solve(l):
            Solves the heat PDE for a given length scale l.
        generate_grf(grid, length_scale, sigma=1.0, random_seed=None):
            Generates a mean-zero Gaussian random field on a given grid.
    """

    def __init__(self, N_x, X, N_t, T, N_v_x, N_v_num, D_const, k, device):     
        self.N_x = N_x
        self.X = X
        self.N_t = N_t
        self.T = T
        self.N_v_x = N_v_x
        self.N_v_num = N_v_num
        self.x = np.linspace(0, self.X, N_x + 1)
        self.x_database = np.linspace(0, self.X, 1000+1)
        self.t = np.linspace(0, self.T, N_t + 1)
        self.x_v = np.linspace(0, self.X, N_v_x + 1)
        self.D_const = D_const
        self.k = k
        self.device = device
    
    def solve(self, l, D_type='const', v_type='nonzero', run_type='train'):
        # parameters
        self.l = l
        self.D_type = D_type
        self.v_type = v_type
        self.run_type = run_type
        
        # initialisation
        u0 = np.zeros((1001))  # Initial condition
        # u0v = np.zeros(self.N_v_x + 1)  # Initial condition
        v_database = np.zeros((self.N_v_num, 1001))  
        D_database = np.zeros((self.N_v_num, 1001))
        ux = np.zeros((self.N_v_num, self.N_x + 1, self.N_t + 1)) # Initialize the solution
        ux_v = np.zeros((self.N_v_num, self.N_v_x + 1, self.N_t + 1)) # Initialize the solution

        # Generate database for the random field and diffusion coefficient
        if self.run_type == 'train':
            seeds = np.arange(self.N_v_num)
        elif self.run_type == 'test':
            seeds = np.arange(10000, 10000 + self.N_v_num)

        v_database = self.generate_v(self.v_type, self.x_database, seeds)
        if self.D_type == 'const':
            D_database = np.tile(self.D_const * np.ones_like(self.x_database), (self.N_v_num, 1))
        else:
            D_database = self.D_const * (1 + np.abs(self.control_grf(self.l, self.x_database, seeds)))

        # Solve the PDE for different random fields
        for i in range(self.N_v_num):
            vi_database = v_database[i, :]
            Di_database = D_database[i, :]
            # Solve the PDE
            solution = solve_ivp(
                self.diff_react_pde, [self.t[0], self.t[-1]], u0, method='RK45', t_eval=self.t, args=(vi_database, Di_database))

            # select self.x from self.x_database for the random field to solve the PDE
            index_x = np.searchsorted(self.x_database, self.x)
            ux[i, :, :] = solution.y[index_x, :].reshape((self.N_x + 1, self.N_t + 1))    # Reshape the solution

            # select self.x_v from self.x_database for the random field to solve the PDE
            index_x_v = np.searchsorted(self.x_database, self.x_v)
            ux_v[i, :, :] = solution.y[index_x_v, :].reshape((self.N_v_x + 1, self.N_t + 1))    # Reshape the solution


        # select v and D for neural ODE
        v = v_database[:, index_x_v]
        D = D_database[:, index_x_v]

        # convert to torch tensors
        ux = torch.tensor(ux, dtype=torch.float32, device=self.device)
        ux_v = torch.tensor(ux_v, dtype=torch.float32, device=self.device)
        v = torch.tensor(v, dtype=torch.float32, device=self.device)
        D = torch.tensor(D, dtype=torch.float32, device=self.device)
        t = torch.tensor(self.t, dtype=torch.float32, device=self.device)
        x = torch.tensor(self.x, dtype=torch.float32, device=self.device).unsqueeze(-1)
        
        return ux, ux_v, v, D, t, x
    
    def diff_react_pde(self, t, u, v, D):
        u = u.reshape((-1,))

        # Define the spatial derivative of u
        dudx = np.zeros_like(u)
        dudx[1:-1] = (u[2:] - u[:-2]) / (self.x[1] - self.x[0]) / 2  # Central difference
        dudx[0] = (u[1] - u[0]) / (self.x[1] - self.x[0])  # Forward difference
        dudx[-1] = (u[-1] - u[-2]) / (self.x[-1] - self.x[-2])  # Backward difference
        dudx = (D * dudx).reshape((-1,))
        dudxx = np.zeros_like(u)
        dudxx[1:-1] = (dudx[2:] - dudx[:-2]) / (self.x[1] - self.x[0]) / 2  # Central difference
        dudxx[0] = (dudx[1] - dudx[0]) / (self.x[1] - self.x[0])  # Forward difference
        dudxx[-1] = (dudx[-1] - dudx[-2]) / (self.x[-1] - self.x[-2])  # Backward difference

        return dudxx + self.k*u*u + v
    

    def generate_v(self, v_type, x, random_seed):
        # Ensure random_seed is an array
        if np.isscalar(random_seed):
            random_seed = np.array([random_seed])
        
        if v_type == 'nonzero':
            # Will return shape (S, len(x))
            v = self.control_grf(self.l, x, random_seed)
        elif v_type == 'zero':
            v = np.zeros((random_seed.size, x.size))
        elif v_type == 'const':
            v = np.ones((random_seed.size, x.size))
        elif v_type == 'function':
            # Replicate the function output for each seed
            v = np.tile(np.sin(2 * np.pi * x), (random_seed.size, 1))
        return v

    def control_grf(self, l, x, random_seed):
        # Prepare grid (shape: (n, 1)) for computation
        grid = np.expand_dims(x, axis=1)
        # generate_grf will now generate S samples (one per seed)
        v = self.generate_grf(grid, l, sigma=1, random_seed=random_seed)
        # v is of shape (S, len(x))
        return v

    def generate_grf(self, grid, length_scale, sigma=1.0, random_seed=None):
        """
        Generate a mean-zero Gaussian random field on a grid.

        Parameters:
            grid (ndarray): Points in the space where the field is defined (n x d array).
            length_scale (float): Length scale of the Gaussian kernel.
            sigma (float): Standard deviation of the field.
            random_seed (int or array-like, optional): Random seed(s) for reproducibility.

        Returns:
            ndarray: Realizations of the Gaussian random field at the grid points.
                    If multiple seeds are provided, the shape is (S, n),
                    where S is the number of seeds.
        """
        # Compute pairwise distances and the covariance matrix (n x n)
        distances = cdist(grid, grid, metric='euclidean')
        covariance_matrix = sigma**2 * np.exp(-distances**2 / (2 * length_scale**2))
        mean = np.zeros(len(grid))
        
        # Ensure random_seed is an array if provided
        if random_seed is None:
            # No seed provided, return one sample
            grf = np.random.multivariate_normal(mean, covariance_matrix)
            grf = grf.reshape((1, -1))  # shape (1, n)
        elif np.isscalar(random_seed):
            # A single seed was provided, wrap it in an array
            rng = np.random.default_rng(random_seed)
            grf = rng.multivariate_normal(mean, covariance_matrix)
            grf = grf.reshape((1, -1))
        else:
            # random_seed is an array-like of seeds: loop over them to get reproducible samples.
            grf_list = []
            for seed in random_seed:
                rng = np.random.default_rng(seed)
                sample = rng.multivariate_normal(mean, covariance_matrix)
                grf_list.append(sample)
            grf = np.stack(grf_list, axis=0)  # shape (S, n)
        return grf

    
def Data_divide(u, t, x, percent_train):
    """
    Divide the data into training and testing sets.

    Parameters:
        u (ndarray): Solution of the PDE.
        t (ndarray): Time points.
        x (ndarray): Spatial points.
        percent_train (float): Percentage of data to be used for training.

    Returns:
        ndarray: Training solution.
        ndarray: Training time points.
        ndarray: Training spatial points.
        ndarray: Testing solution.
        ndarray: Testing time points.
        ndarray: Testing spatial points.
    """
    N_t = u.shape[1]
    N_x = u.shape[0]
    N_train = int(percent_train * N_x) # Number of training points
    N_test = N_x - N_train
    # u_train is randomly selected from u
    idx_train = np.random.choice(N_x, N_train, replace=False)
    idx_test = np.setdiff1d(np.arange(N_x), idx_train)
    # Divide the data into training and testing
    u_train = u[idx_train, :, :]
    t_train = t
    x_train = x[idx_train, :]
    u_test = u[idx_test, :, :]
    t_test = t
    x_test = x[idx_test, :]

    return u_train, t_train, x_train, u_test, t_test, x_test
'''