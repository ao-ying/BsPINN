import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import time
import torch
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from torch.nn.parameter import Parameter
import csv

# System settings
torch.set_default_dtype(torch.float)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Device configuration
path = os.path.dirname(__file__) + "/"
torch.backends.cudnn.benchmark = True
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 20

# Exact solution
def solution(x):
    if len(x.shape) > 1: # x is an n*d vector
        temp = np.sin(np.pi / 2 * x)
        out = np.sum(temp, axis=1)
    else : # x is a d-dimensional vector
        temp = np.sin(np.pi / 2 * x)
        out = np.sum(temp)
    return out

# Calculate partial derivative
def grad(f, x):
    ret = torch.autograd.grad(f, x, torch.ones_like(f).to(device), retain_graph=True, create_graph=True)[0]
    return ret

# Output model parameter information
def get_parameter_number(net):
    trainable_num = sum(p.numel() for p in net.params if p.requires_grad)
    return trainable_num

# Generate dataset
def load_data(num_domain, num_boundary, num_test, lb_geom, ub_geom, dim_x):
    # Generate training samples
    X_train = np.random.uniform(lb_geom, ub_geom, (num_domain, dim_x))
    ## Boundary
    num_per_dim = int(num_boundary / dim_x)
    tool = np.zeros(num_per_dim)
    tool[:int(num_per_dim/2)] = ub_geom
    tool[int(num_per_dim/2):] = lb_geom
    for i in range(dim_x):
        boundary_points = np.random.uniform(lb_geom, ub_geom, (num_per_dim, dim_x))
        boundary_points[:, i] = tool
        X_train = np.r_[X_train, boundary_points]

    # Test set
    X_test = np.random.uniform(lb_geom, ub_geom, (num_test, dim_x))
    Y_test = solution(X_test)
    Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))

    return X_train, X_test, Y_test


class PINN(nn.Module):
    def __init__(self, Layers, num_domain, boundary_weight, lb_X, ub_X, dim_x):
        super(PINN, self).__init__()
        # Initialize parameters
        self.lb_X = lb_X
        self.ub_X = ub_X
        self.dim_x = dim_x
        self.Layers = Layers
        self.num_Layers = len(self.Layers)
        self.num_domain = num_domain
        self.boundary_weight = boundary_weight
        self.params = []
        self.act = torch.sin
        # Binary neural network related
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
        # Middle hidden layers
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
    
    # Binary neural network part
    def neural_net(self, X):
        # Data preprocessing
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
        # Equation
        X_e = [0 for i in range(self.dim_x)]
        for i in range(self.dim_x):
            X_e[i] = X[0:self.num_domain, i : i + 1].clone()
            X_e[i] = X_e[i].requires_grad_()
        u_e = self.neural_net(torch.cat(X_e, dim = 1))
        dudx = [grad(u_e, X_e[i]) for i in range(self.dim_x)]
        dudx2 = [grad(dudx[i], X_e[i]) for i in range(self.dim_x)]
        dudx2 = torch.cat(dudx2, dim=1) # self.num_domain * dim_x
        Laplace_u = torch.sum(dudx2, dim=1, keepdim=True)  # self.num_domain * 1
        temp_sin = [torch.sin(np.pi / 2 * X_e[i]) for i in range(self.dim_x)]
        temp_sin = torch.cat(temp_sin, dim=1)
        f = np.pi**2 / 4 * torch.sum(temp_sin, dim=1, keepdim=True)
        equation = - Laplace_u - f

        # Boundary conditions
        x_b = X[self.num_domain:].clone()
        u_b = self.neural_net(x_b)
        temp_sin = torch.sin(np.pi / 2 * x_b)
        dvalue = torch.sum(temp_sin, dim=1, keepdim=True)
        boundary = u_b - dvalue

        return equation, boundary

    # Loss function
    def loss(self, X_train):
        # Calculate equation and boundary condition terms
        equation, boundary = self.he_net(X_train)
        
        # Calculate total error
        loss_e = torch.mean(torch.square(equation))
        loss_b = torch.mean(torch.square(boundary))
        loss_all = loss_e + self.boundary_weight * loss_b

        return loss_all
    
    # Calculate relative error
    def rel_error(self):
        u_pred = self.neural_net(X_test).cpu().detach().numpy()
        u_truth = Y_test.cpu().detach().numpy()
        u_error = np.linalg.norm((u_truth - u_pred), ord=2) / (np.linalg.norm(u_truth, ord=2))
        return u_error
    
    # Predict u value at X
    def predict(self, X):
        u_pred = self.neural_net(X)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred


# main function
if __name__ == "__main__":
    seeds = [11]
    csv_data = []
    for seed in seeds:
        # Random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Parameter settings
        ## Neural network related
        train = True
        name = "poisson5d"
        epochs = 20000 # Number of iterations for adam optimizer
        num_domain = 8000 # Number of training points inside the solution domain
        num_boundary = 4000 # Number of training points on the boundary, must be divisible by dim_x
        num_test = 500000 
        Layers = [5, 96, 48, 24, 12, 6, 1]
        learning_rate = 0.001 # Learning rate, auto-decay strategy to be set
        shuffle = False # Whether to shuffle the data at the start of each epoch
        boundary_weight = 100
        patience = max(10, epochs / 10)
        weight_decay = 0
        
        ## Equation related
        lb_geom = 0 # Lower bound of x
        ub_geom = 1 # Upper bound of x
        dim_x = 5 # Dimension of x
        dim_y = 1 # Dimension of y
        
        ## Plot related
        align = False 
        lb_loss = 1e-2
        ub_loss = 1e6
        lb_u = -1.7
        ub_u = 1.7
        lb_error = 3e-2
        ub_error = 2e0
        error_param = 5 # Plot (20 * error_param) error points, epochs must be divisible by (20 * error_param).
        
        # Auxiliary variables
        name = name + ("_%d" % epochs)
        print("\n\n***** name = %s *****" % name)
        print("seed = %d" % seed)
        output_path = path + ('./output_BsPINN')
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('./output_BsPINN/%s/' % name)
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('./output_BsPINN/%s/train_%d/' % (name, seed))
        if not os.path.exists(output_path): os.mkdir(output_path)

        # Generate dataset
        if train:
            print("Loading data.")
            X_train, X_test, Y_test = load_data(num_domain, num_boundary, num_test, lb_geom, ub_geom, dim_x)
            lb_X = X_train.min(0)
            ub_X = X_train.max(0)
            lb_X = torch.from_numpy(lb_X).float().to(device)
            ub_X = torch.from_numpy(ub_X).float().to(device)
            X_train = torch.from_numpy(X_train).float().to(device)
            X_test = torch.from_numpy(X_test).float().to(device)
            Y_test = torch.from_numpy(Y_test).float().to(device)

            # Declare neural network instance
            model = PINN(Layers, num_domain, boundary_weight, lb_X, ub_X, dim_x)
            model = nn.DataParallel(model)
            model = model.module
            model.to(device)
            print(model) # Print network summary
            params = model.num_param # Output parameter information
            print("Total number of parameters: %d" % params)
            
            # Adam optimizer
            optimizer = optim.Adam(model.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, verbose=True, min_lr=1e-6)

            # Training
            start = time.time()
            loss_list = []
            error_list = []
            min_loss = 9999999
            print("Start training")
            for it in range(epochs):
                # Optimizer training
                loss = model.loss(X_train) 
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                # Decay learning rate
                scheduler.step(loss)

                # Save loss value
                loss_val = loss.cpu().detach().numpy()
                loss_list.append(loss_val)

                # Save the model with the minimum training loss
                if loss_val < min_loss:
                    torch.save(model, output_path + 'network.pkl')
                    min_loss = loss_val
                    
                # Save error curve
                if (it + 1) % (epochs / 20 / error_param) == 0:
                    u_L2RE = model.rel_error()
                    error_list.append(u_L2RE)

                # Output
                if (it + 1) % (epochs / 20) == 0:
                    print("It = %d, loss = " % (it + 1), loss_val, ", u_L2RE = ", u_L2RE, ", finish: %d%%" % ((it + 1) / epochs * 100))
                    
            end = time.time()
            print("Total train time: %.2fs" % (end - start))
            loss_list = np.array(loss_list).flatten()
            np.savetxt(output_path + "loss.txt", loss_list, fmt="%s", delimiter=' ')
            min_loss = np.min(loss_list)
            np.savetxt(output_path + "error.txt", error_list, fmt="%s", delimiter=' ')
            print("Min train loss: %.8f" % min_loss)
        else:
            X_train, X_test, Y_test = load_data(num_domain, num_boundary, num_test, lb_geom, ub_geom, dim_x)
            X_test = torch.from_numpy(X_test).float().to(device)
            Y_test = torch.from_numpy(Y_test).float().to(device)
        torch.cuda.empty_cache() # Release GPU memory
        
        # Save loss curve
        loss_list = np.loadtxt(output_path + "loss.txt", dtype = float, delimiter=' ')
        fig, ax = plt.subplots()
        plt.semilogy(loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if align:
            plt.ylim(lb_loss, ub_loss) # Align scale with BiPINN
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
            plt.ylim(lb_error, ub_error) # Align scale with BiPINN
            plt.savefig(output_path + 'error_aligned.pdf', format="pdf", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(output_path + 'error.pdf', format="pdf", dpi=300, bbox_inches="tight")

        # Calculate error
        u_truth = Y_test.cpu().detach().numpy().flatten()
        model2 = torch.load(output_path + 'network.pkl', map_location=device) 
        u_pred = model2.predict(X_test).flatten()
        u_L2RE = np.linalg.norm((u_truth - u_pred), ord=2) / (np.linalg.norm(u_truth, ord=2))
        print("u_L2RE = %.8f" % u_L2RE)
        
        # Save error value
        csv_data.append([seed, u_L2RE])
    
    # Save as csv file
    output_path = path + ('./output_BsPINN/%s/' % name)
    file_write = open(output_path + 'record.csv', 'w')
    writer = csv.writer(file_write)
    writer.writerow(['seed', 'L2RE'])
    writer.writerows(csv_data)
    file_write.close()
