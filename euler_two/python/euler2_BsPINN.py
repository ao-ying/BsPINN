import os
import csv
import numpy as np
from matplotlib import pyplot as plt
import time
import torch
import torch.nn as nn         
import torch.optim as optim             
from torch.nn.parameter import Parameter

# System settings
torch.set_default_dtype(torch.float)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Device configuration
path = os.path.dirname(__file__) + "/"
torch.backends.cudnn.benchmark = True
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 20

# Compute partial derivative
def grad(f, x):
    ret = torch.autograd.grad(f, x, torch.ones_like(f).to(device), retain_graph=True, create_graph=True)[0]
    return ret

# Output model parameter information
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.params)
    trainable_num = sum(p.numel() for p in net.params if p.requires_grad)
    return trainable_num
    
# Set the learning rate of the optimizer to lr
def reset_lr(optimizer, lr):
    for params in optimizer.param_groups:
        params['lr'] = lr

# Cumulative sum of elements in the list
def accsum(l):
    ret = [l[0]]
    temp = l[0]
    for i in range(1, len(l)):
        temp += l[i]
        ret.append(temp)
    return ret

# Exact solution for density
def sol_r(x, y, t):
    ret = 0
    if len(x.shape) == 0:
        if x <= (0.5 + 0.1 * t):
            ret = 1.4
        else:
            ret = 1.0
    else:
        ret = []
        for i in range(x.shape[0]):
            if x[i] <= (0.5 + 0.1 * t[i]):
                ret.append(1.4)
            else:
                ret.append(1.0)
        ret = np.array(ret)
    return ret

# Construct training and testing sets for the domain, initial condition, and boundary conditions
# N0: number of initial condition training points, Nb: number of boundary condition training points, Nf: number of domain interior training points
def load_data(N0, Nb, Nf, lb_x, ub_x, lb_y, ub_y, lb_t, ub_t, dim_test):
    # Training set: interior of the domain
    X_train = np.random.uniform([lb_x, lb_y, lb_t], [ub_x, ub_y, ub_t], (Nf, 3))
    # Training set: initial condition
    points = np.random.uniform([lb_x, lb_y, lb_t], [ub_x, ub_y, lb_t], (N0, 3))
    X_train = np.r_[X_train, points]
    # Training set: boundary conditions
    num_per_edge = int(Nb / 4)
    points = np.random.uniform([lb_x, lb_y, lb_t], [lb_x, ub_y, ub_t], (num_per_edge, 3)) # Left
    X_train = np.r_[X_train, points]
    points = np.random.uniform([ub_x, lb_y, lb_t], [ub_x, ub_y, ub_t], (num_per_edge, 3)) # Right
    X_train = np.r_[X_train, points]
    points = np.random.uniform([lb_x, lb_y, lb_t], [ub_x, lb_y, ub_t], (num_per_edge, 3)) # Bottom
    X_train = np.r_[X_train, points]
    points = np.random.uniform([lb_x, ub_y, lb_t], [ub_x, ub_y, ub_t], (num_per_edge, 3)) # Top
    X_train = np.r_[X_train, points]
    
    # Compute true values (for initial and boundary values)
    r_train = sol_r(X_train[:, 0], X_train[:, 1], X_train[:, 2])
    r_train = np.reshape(r_train, (r_train.shape[0], 1))
    
    # Test set
    dim_test = 100
    X_test = np.loadtxt(path + "../data/points_%d.txt" % dim_test, dtype=float, delimiter=' ')
    Y_test = sol_r(X_test[:, 0], X_test[:, 1], X_test[:, 2])
    
    return X_train, r_train, X_test, Y_test


# BsPINN neural network
class BsPINN(nn.Module):
    def __init__(self, Layers, N0, Nb, Nf, initial_weight, boundary_weight, lb_X, ub_X, gamma, r_train):
        super(BsPINN, self).__init__()
        # Initialize parameters
        self.iter = 0
        self.lb_X = lb_X
        self.ub_X = ub_X
        self.Layers = Layers
        self.num_Layers = len(self.Layers)
        self.r_train = r_train
        self.gamma = gamma
        self.Nf = Nf
        self.N0 = N0
        self.Nb = Nb
        self.act = torch.tanh
        self.lens = accsum([Nf, N0, int(Nb / 4), int(Nb / 4), int(Nb / 4), int(Nb / 4)])
        self.initial_weight = initial_weight
        self.boundary_weight = boundary_weight
        self.params = []
        # Initialize binary neural network
        self.width = [Layers[0]] + [int(pow(2, i - 1) * Layers[i]) for i in range(1, len(Layers) - 1)] + [Layers[-1]]
        self.masks = self.construct_mask()
        self.num_param = self.cal_param()
        self.weights, self.biases = self.initialize_NN(self.Layers)
        
    # Calculate the number of parameters
    def cal_param(self):
        ret = 0
        for i in range(self.num_Layers - 2):
            temp = pow(2, i) * (self.Layers[i] * self.Layers[i + 1] + self.Layers[i + 1])
            ret += temp
        ret += (self.width[-2] * self.Layers[-1] + self.Layers[-1])
        return ret
    
    # Create mask
    def construct_mask(self):
        masks = []
        for l in range(2, self.num_Layers - 2):
            # Calculate block matrix dimensions
            num_blocks = int(pow(2, l - 1))
            blocksize1 = int(self.width[l] / num_blocks)
            blocksize2 = 2 * self.Layers[l + 1]
            blocks = [torch.ones((blocksize1, blocksize2)) for i in range(num_blocks)]
            mask = torch.block_diag(*blocks).to(device)
            masks.append(mask)
        return masks
        
    # Initialize binary neural network parameters
    def initialize_NN(self, Layers):
        weights = []
        biases = []
        # First hidden layer
        tempw = torch.zeros(self.Layers[0], self.width[1]).to(device)
        w = Parameter(tempw, requires_grad=True)
        nn.init.xavier_uniform_(w, gain=1)
        tempb = torch.zeros(1, self.width[1]).to(device)
        b = Parameter(tempb, requires_grad=True)
        weights.append(w)
        biases.append(b)
        self.params.append(w)
        self.params.append(b)
        # Intermediate hidden layers
        for l in range(1, self.num_Layers - 2):
            # Weights w
            tempw = torch.zeros(self.width[l], self.width[l + 1]).to(device)
            # Block diagonal matrix initialization
            for i in range(int(pow(2, l))): # Traverse each small matrix and initialize
                tempw2 = torch.zeros(Layers[l], Layers[l + 1])
                w2 = Parameter(tempw2, requires_grad=True)
                nn.init.xavier_uniform_(w2, gain=1)
                row_index = int(i / 2)
                tempw[row_index * Layers[l] : (row_index + 1) * Layers[l], i * Layers[l + 1] : (i + 1) * Layers[l + 1]] = w2.data
            w = Parameter(tempw, requires_grad=True)
            # Bias b
            tempb = torch.zeros(1, self.width[l + 1]).to(device)
            b = Parameter(tempb, requires_grad=True)
            weights.append(w)
            biases.append(b)
            self.params.append(w)
            self.params.append(b)
        # Last hidden layer
        tempw = torch.zeros(self.width[-2], self.Layers[-1]).to(device)
        w = Parameter(tempw, requires_grad=True)
        tempb = torch.zeros(1, self.Layers[-1]).to(device)
        b = Parameter(tempb, requires_grad=True)
        weights.append(w)
        biases.append(b)
        self.params.append(w)
        self.params.append(b)
        return weights, biases
    
    # Binary neural network part
    def neural_net(self, X):
        # Data preprocessing, lb and ub used during training and testing should be the same, otherwise the neural networks used for training and testing will be different.
        X = 2.0 * (X - self.lb_X) / (self.ub_X - self.lb_X) - 1.0
        X = X.float()
        
        # Network part
        for l in range(0, self.num_Layers - 2):
            if l >= 2 and l <= self.num_Layers - 3:
                W = self.weights[l]
                W2 = W * self.masks[l - 2]
                b = self.biases[l]
                X = self.act(torch.add(torch.matmul(X, W2), b))
            else:
                W = self.weights[l]
                b = self.biases[l]
                X = self.act(torch.add(torch.matmul(X, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        X = torch.add(torch.matmul(X, W), b)
        return X

    # PDE part
    def he_net(self, X):
        # Equations
        x_e = X[0 : self.lens[0], 0:1].clone()
        x_e = x_e.requires_grad_()
        y_e = X[0 : self.lens[0], 1:2].clone()
        y_e = y_e.requires_grad_()
        t_e = X[0 : self.lens[0], 2:3].clone()
        t_e = t_e.requires_grad_()
        out_e = self.neural_net(torch.cat([x_e, y_e, t_e], dim = 1))
        r_e = out_e[:, 0:1]
        u_e = out_e[:, 1:2]
        v_e = out_e[:, 2:3]
        p_e = out_e[:, 3:4]
        E_e = out_e[:, 4:5]
        equation1 = grad(r_e, t_e) + grad(r_e * u_e, x_e) + grad(r_e * v_e, y_e)
        equation2 = grad(r_e * u_e, t_e) + grad(r_e * u_e**2 + p_e, x_e) + grad(r_e * u_e * v_e, y_e)
        equation3 = grad(r_e * v_e, t_e) + grad(r_e * u_e * v_e, x_e) + grad(r_e * v_e**2 + p_e, y_e)
        equation4 = grad(r_e * E_e, t_e) + grad(u_e * (r_e * E_e + p_e), x_e) + grad(v_e * (r_e * E_e + p_e), y_e)
        equation5 = p_e - (self.gamma - 1) * (r_e * E_e - 0.5 * r_e * (u_e**2 + v_e**2))
        equations = [equation1, equation2, equation3, equation4, equation5]
        
        # Initial conditions
        initials = []
        out_i1 = self.neural_net(X[self.lens[0]: self.lens[1], :])
        rval = self.r_train[self.lens[0]: self.lens[1]]
        r_i = out_i1[:, 0:1]
        u_i = out_i1[:, 1:2]
        v_i = out_i1[:, 2:3]
        p_i = out_i1[:, 3:4]
        initials.extend([r_i - rval, u_i - 0.1, v_i - 0, p_i - 1.0])
        
        # Boundary conditions
        boundarys = []
        # Left
        out_b1 = self.neural_net(X[self.lens[1]: self.lens[2], :])
        rval = self.r_train[self.lens[1]: self.lens[2]]
        r_b = out_b1[:, 0:1]
        u_b = out_b1[:, 1:2]
        v_b = out_b1[:, 2:3]
        p_b = out_b1[:, 3:4]
        boundarys.extend([r_b - rval, u_b - 0.1, v_b - 0, p_b - 1.0])
        # Right
        out_b2 = self.neural_net(X[self.lens[2]: self.lens[3], :])
        rval = self.r_train[self.lens[2]: self.lens[3]]
        r_b = out_b2[:, 0:1]
        u_b = out_b2[:, 1:2]
        v_b = out_b2[:, 2:3]
        p_b = out_b2[:, 3:4]
        boundarys.extend([r_b - rval, u_b - 0.1, v_b - 0, p_b - 1.0])
        # Top
        out_b3 = self.neural_net(X[self.lens[3]: self.lens[4], :])
        rval = self.r_train[self.lens[3]: self.lens[4]]
        r_b = out_b3[:, 0:1]
        u_b = out_b3[:, 1:2]
        v_b = out_b3[:, 2:3]
        p_b = out_b3[:, 3:4]
        boundarys.extend([r_b - rval, u_b - 0.1, v_b - 0, p_b - 1.0])
        # Bottom
        out_b4 = self.neural_net(X[self.lens[4]: self.lens[5], :])
        rval = self.r_train[self.lens[4]: self.lens[5]]
        r_b = out_b4[:, 0:1]
        u_b = out_b4[:, 1:2]
        v_b = out_b4[:, 2:3]
        p_b = out_b4[:, 3:4]
        boundarys.extend([r_b - rval, u_b - 0.1, v_b - 0, p_b - 1.0])
        
        return equations, initials, boundarys

    # Loss function
    def loss(self, X_train):
        # Compute equation and boundary terms
        equations, initials, boundarys = self.he_net(X_train)
        
        # Compute total error
        loss_e = 0
        for i in range(len(equations)): loss_e += torch.mean(torch.pow(equations[i], 2))
        loss_i = 0
        for i in range(len(initials)): loss_i += torch.mean(torch.pow(initials[i], 2))
        loss_b = 0
        for i in range(len(boundarys)): loss_b += torch.mean(torch.pow(boundarys[i], 2))
        loss_all = loss_e + self.initial_weight * loss_i + self.boundary_weight * loss_b

        return loss_all

    # Predict the value of u at point X
    def predict(self, X):
        pred = self.neural_net(X)
        pred = pred.cpu().detach().numpy()
        return pred

    # Calculate relative error
    def rel_error(self):
        pred = self.predict(X_test)
        r_pred = pred[:, 0]
        r_truth = Y_test
        r_L2RE = np.linalg.norm((r_truth - r_pred), ord=2) / (np.linalg.norm(r_truth, ord=2))
        return r_L2RE


# main function
if __name__ == "__main__":
    seeds = [13]
    csv_data = []
    for seed in seeds:
        print("***** seed = %d *****" % seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Parameter settings
        ## Equation related
        gamma = 1.4
        lb_x = 0
        ub_x = 1
        lb_y = 0
        ub_y = 1
        lb_t = 0
        ub_t = 2
        
        ## Neural network related
        train = True
        name = "256-16_v9"
        epochs = 20000 # Number of iterations for the adam optimizer
        Nf = 30000 # Number of training points inside the solution domain
        N0 = 400 # Number of initial condition training points
        Nb = 400 # Number of boundary condition training points
        Layers = [3, 256, 128, 64, 32, 16, 5] # Fully connected network structure
        learning_rate = 0.001 # Initial learning rate
        patience = 5000 # Patience for plateau learning rate decay
        initial_weight = 1 # Initial condition loss weight
        boundary_weight = 1 # Boundary condition loss weight
        weight_decay = 0 # 0.0001
        dim_test = 100
        
        ## Plotting related
        align = False
        lb_loss = 1e-5
        ub_loss = 5e0
        lb_r = 0.9
        ub_r = 1.5
        lb_error = 8e-3
        ub_error = 0.3
        lb_r_error = -0.25
        ub_r_error = 0.25
        error_param = 1 # 5 # Plot (20 * error_param) error points, epochs need to be divisible by (20 * error_param).
        
        # Auxiliary variables
        name = name + ("_%d" % epochs)
        print("\n\n***** name = %s *****" % name)
        print("seed = %d" % seed)
        output_path = path + './output_BsPINN'
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + './output_BsPINN/%s/' % name
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('./output_BsPINN/%s/train_%d/' % (name, seed))
        if not os.path.exists(output_path): os.mkdir(output_path)

        # Generate dataset
        if train:
            print("Loading data")
            X_train, r_train, X_test, Y_test = load_data(N0, Nb, Nf, lb_x, ub_x, lb_y, ub_y, lb_t, ub_t, dim_test)
            
            # Data processing
            lb_X = X_train.min(0) # Get the minimum value of each dimension of the training set
            ub_X = X_train.max(0) # Get the maximum value of each dimension of the training set
            lb_X = torch.from_numpy(lb_X).float().to(device)
            ub_X = torch.from_numpy(ub_X).float().to(device)
            X_train = torch.from_numpy(X_train).float().to(device)
            r_train = torch.from_numpy(r_train).float().to(device)
            X_test = torch.from_numpy(X_test).float().to(device)
            
            # Declare neural network instance
            model = BsPINN(Layers, N0, Nb, Nf, initial_weight, boundary_weight, lb_X, ub_X, gamma, r_train)
            model = nn.DataParallel(model)
            model = model.module
            model.to(device)
            print(model) # Print network summary
            params = model.num_param
            print("params = %d" % params)
            
            # Adam optimizer
            optimizer = optim.Adam(model.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, verbose=True, min_lr=1e-6) # Plateau learning rate decay

            # Training
            start = time.time()
            ## adam
            loss_list = []
            error_list = []
            min_loss = 9999999999
            print("Start Adam training!")
            for it in range(epochs):
                # Optimizer training
                loss = model.loss(X_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Learning rate decay
                scheduler.step(loss)

                # Save loss values and relative errors, and save the model with the minimum training error
                loss_val = loss.cpu().detach().numpy() # If running on GPU, data needs to be transferred to CPU first
                loss_list.append(loss_val)
                
                # Save the model with the minimum train loss
                if loss_val < min_loss:
                    torch.save(model, output_path + 'network.pkl')
                    min_loss = loss_val
                    
                # Save error curve
                if (it + 1) % (epochs / 20 / error_param) == 0:
                    u_L2RE = model.rel_error()
                    error_list.append(u_L2RE)

                # Output
                if (it + 1) % (epochs / 20) == 0:
                    # Output
                    print("It = %d, loss = %.8f, r_L2RE = %.8f, finish: %d%%" % ((it + 1), loss_val, u_L2RE, ((it + 1) / epochs * 100)))
                
            # Post-processing
            end = time.time()
            train_time = end - start
            loss_list = np.array(loss_list).flatten()
            min_loss = np.min(loss_list)
            np.savetxt(output_path + "/loss.txt", loss_list, fmt="%s", delimiter=' ')
            np.savetxt(output_path + "error.txt", error_list, fmt="%s", delimiter=' ')
            print("time = %.2fs" % train_time)
            print("min_loss = %.8f" % min_loss)

        # Save loss curve
        loss_list = np.loadtxt(output_path + "/loss.txt", dtype = float, delimiter=' ')
        fig, ax = plt.subplots()
        plt.semilogy(loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if align:
            plt.ylim(lb_loss, ub_loss) # Same scale as BiPINN
            plt.savefig(output_path + 'loss_aligned.pdf', format="pdf", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(output_path + 'loss.pdf', format="pdf", dpi=300, bbox_inches="tight")
            
        # Save error curve
        fig, ax = plt.subplots()
        error_list = np.loadtxt(output_path + "error.txt", dtype = float, delimiter=' ')
        plt.semilogy(error_list)
        plt.xlabel('Epoch')
        plt.ylabel('Relative error')
        tool = [0, 4, 8, 12, 16, 20]
        plt.xticks([tool[i] * error_param for i in range(len(tool))], [str(0), str(int(epochs * 0.2)), str(int(epochs * 0.4)), str(int(epochs * 0.6)), str(int(epochs * 0.8)), str(int(epochs))])
        if align:
            plt.ylim(lb_error, ub_error) # Same scale as BiPINN
            plt.savefig(output_path + 'error_aligned.pdf', format="pdf", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(output_path + 'error.pdf', format="pdf", dpi=300, bbox_inches="tight")
            
        # Error evaluation
        dim = dim_test
        points = np.loadtxt(path + "../data/points_%d.txt" % dim, dtype = float, delimiter=' ')
        X_pred = torch.from_numpy(points).float().to(device)
        model2 = torch.load(output_path + 'network.pkl', map_location=device)
        pred = model2.predict(X_pred)
        r_pred = pred[:, 0]
        r_truth = sol_r(points[:, 0], points[:, 1], points[:, 2])
        r_L2RE = np.linalg.norm((r_truth - r_pred), ord=2) / (np.linalg.norm(r_truth, ord=2))
        print("r_L2RE = %.8f" % r_L2RE)

        # Prepare data for plotting
        dim = 500
        X = np.linspace(lb_x, ub_x, dim)
        Y = np.linspace(lb_y, ub_y, dim)
        X, Y = np.meshgrid(X, Y)
        X = X.flatten().reshape(dim * dim, 1)
        Y = Y.flatten().reshape(dim * dim, 1)
        T = np.full((dim * dim, 1), 1.0)
        X_pred = np.c_[X, Y, T] # N * 3
        X_pred = torch.from_numpy(X_pred).float().to(device)
        model2 = torch.load(output_path + 'network.pkl', map_location=device)
        pred = model2.predict(X_pred)
        r_pred = pred[:, 0]
        r_truth = sol_r(X.flatten(), Y.flatten(), T.flatten())

        # Plot predicted solution r
        fig, ax = plt.subplots()
        if align:
            levels = np.arange(lb_r, ub_r, (ub_r - lb_r) / 100)
        else:
            levels = np.arange(min(r_pred) - abs(max(r_pred) - min(r_pred)) / 10, max(r_pred) + abs(max(r_pred) - min(r_pred)) / 10, (max(r_pred) - min(r_pred)) / 100)
        cs = ax.contourf(X.reshape(dim, dim), Y.reshape(dim, dim), r_pred.reshape(dim, dim), levels, cmap=plt.get_cmap('Spectral'))
        cbar = fig.colorbar(cs)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('$r$(BsPINN) when t=1s')
        if align:
            plt.savefig(output_path + "r_pred_aligned.png", format="png", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(output_path + "r_pred.png", format="png", dpi=300, bbox_inches="tight")
            
        # Plot true solution r
        fig, ax = plt.subplots()
        if align:
            levels = np.arange(lb_r, ub_r, (ub_r - lb_r) / 100)
        else:
            levels = np.arange(min(r_truth) - abs(max(r_truth) - min(r_truth)) / 10, max(r_truth) + abs(max(r_truth) - min(r_truth)) / 10, (max(r_truth) - min(r_truth)) / 100)
        cs = ax.contourf(X.reshape(dim, dim), Y.reshape(dim, dim), r_truth.reshape(dim, dim), levels, cmap=plt.get_cmap('Spectral'))
        cbar = fig.colorbar(cs)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('$r$(truth) when t=1s')
        if align:
            plt.savefig(output_path + "r_truth_aligned.png", format="png", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(output_path + "r_truth.png", format="png", dpi=300, bbox_inches="tight")
            
        # Plot error image r
        fig, ax = plt.subplots()
        r_diff = r_truth - r_pred
        if align:
            levels = np.arange(lb_r_error, ub_r_error, (ub_r_error - lb_r_error) / 100)
        else:
            levels = np.arange(min(r_diff) - abs(max(r_diff) - min(r_diff)) / 10, max(r_diff) + abs(max(r_diff) - min(r_diff)) / 10, (max(r_diff) - min(r_diff)) / 100)
        cs = ax.contourf(X.reshape(dim, dim), Y.reshape(dim, dim), r_diff.reshape(dim, dim), levels, cmap=plt.get_cmap('Spectral'))
        cbar = fig.colorbar(cs)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('$r$(difference) when t=1s')
        if align:
            plt.savefig(output_path + "r_diff_aligned.png", format="png", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(output_path + "r_diff.png", format="png", dpi=300, bbox_inches="tight")
        
        # Save error values
        csv_data.append([seed, r_L2RE])
    
    # Save as csv file
    output_path = path + ('./output_BsPINN/%s/' % name)
    file_write = open(output_path + 'record.csv', 'w')
    writer = csv.writer(file_write)
    writer.writerow(['seed', 'L2RE'])
    writer.writerows(csv_data)
    file_write.close()
