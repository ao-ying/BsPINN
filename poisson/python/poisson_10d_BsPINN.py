import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import time
import torch
import torch.nn as nn         
import torch.optim as optim             
from scipy import interpolate
from torch.nn.parameter import Parameter
import copy
import csv

# System settings
torch.set_default_dtype(torch.float)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Device configuration
device_cpu = torch.device('cpu')
path = os.path.dirname(__file__) + "/"
torch.backends.cudnn.benchmark = True
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 30

# Exact solution
def solution(x):
    if len(x.shape) > 1: 
        x_sum = np.sum(x, axis=1)
        x_mean = np.mean(x, axis=1)
    else :
        x_sum = np.sum(x) 
        x_mean = np.mean(x)
    ustar = x_mean ** 2 + np.sin(coe * x_sum) 
    return ustar

# Compute gradient
def grad(f,x):
    ret = torch.autograd.grad(f, x, torch.ones_like(f).to(device), retain_graph=True, create_graph=True)[0]
    return ret

# Output model parameter information
def get_parameter_number(net):
    trainable_num = sum(p.numel() for p in net.params if p.requires_grad)
    return trainable_num
    
#  set learning rate
def reset_lr(optimizer, lr):
    for params in optimizer.param_groups: 
            params['lr'] = lr

# Generate dataset
def load_data(lb_geom, ub_geom, num_sample, num_boundary, dim_x):
    # train set
    ## points within the domain
    X_train = np.random.uniform(lb_geom, ub_geom, (num_sample, dim_x))
    ## boundary
    num_per_dim = int(num_boundary / dim_x)
    tool = np.zeros(num_per_dim)
    tool[:int(num_per_dim/2)] = ub_geom
    tool[int(num_per_dim/2):] = lb_geom
    for i in range(dim_x):
        boundary_points = np.random.uniform(lb_geom, ub_geom, (num_per_dim, dim_x))
        boundary_points[:, i] = tool
        X_train = np.r_[X_train, boundary_points]
    
    return X_train

# Residual Block
class Residual(nn.Module):
    def __init__(self, Layers):
        super().__init__()
        # Data initialization
        self.Layers = copy.deepcopy(Layers)
        self.Layers.insert(0, Layers[0])  # One less layer compared to a regular binary neural network.
        self.num_Layers = len(self.Layers)
        self.width = [self.Layers[0]] + [int(pow(2, i - 1) * self.Layers[i]) for i in range(1, len(self.Layers))]  # One less layer compared to a regular binary neural network.
        self.masks = self.construct_mask()
        self.num_params = self.cal_param()
        
        # Weight initialization
        self.params = []
        # Initialize binary neural network
        self.weights, self.biases =  self.initialize_NN(self.Layers)
        
    # Calculate number of parameters
    def cal_param(self):
        ret = 0
        for i in range(self.num_Layers - 1):
            temp = pow(2, i) * (self.Layers[i] * self.Layers[i + 1] + self.Layers[i + 1])
            ret += temp
        return ret
    
    # Create masks
    def construct_mask(self):
        masks = []
        for l in range(2, self.num_Layers - 1):
            # Calculate block matrix dimensions
            num_blocks = int(pow(2, l - 1))
            blocksize1 = int(self.width[l] / num_blocks)
            blocksize2 = 2 * self.Layers[l + 1]
            blocks = [torch.ones((blocksize1,blocksize2)) for i in range(num_blocks)]
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
        for l in range(1,self.num_Layers - 1):
            # Weight w
            tempw = torch.zeros(self.width[l], self.width[l+1]).to(device)
            # Block diagonal matrix initialization
            for i in range(int(pow(2,l))):  # Iterate over each small matrix and initialize
                tempw2 = torch.zeros(Layers[l], Layers[l+1])
                w2 = Parameter(tempw2, requires_grad=False)
                nn.init.xavier_uniform_(w2, gain=1)  
                row_index = int(i / 2)
                tempw[row_index * Layers[l] : (row_index + 1) * Layers[l], i * Layers[l+1] : (i + 1) * Layers[l+1]] = w2.data
            w = Parameter(tempw, requires_grad=True)
            # Bias b
            tempb = torch.zeros(1, self.width[l+1]).to(device)
            b = Parameter(tempb, requires_grad=True)
            weights.append(w)
            biases.append(b) 
            self.params.append(w)
            self.params.append(b)
        return weights, biases
        
    def forward(self, X):
        H = X
        # Neural network part
        for l in range(0, self.num_Layers - 1):
            if l >=2 and l <= self.num_Layers - 2:
                W = self.weights[l]
                W2 = W * self.masks[l - 2]
                b = self.biases[l]
                H = torch.add(torch.matmul(H, W2), b)
                if l != self.num_Layers - 2:
                    H = torch.sin(H)
            else:
                W = self.weights[l]
                b = self.biases[l]
                H = torch.add(torch.matmul(H, W), b)
                H = torch.sin(H) 
        # Residual sum  
        Y = torch.sin(torch.add(H, X))
        return Y
    
    def forward2(self, X):
        H = X
        # Neural network part
        for l in range(0, self.num_Layers - 1):
            if l >=2 and l <= self.num_Layers - 2:
                W = self.weights2[l]
                W2 = W * self.masks2[l - 2]
                b = self.biases2[l]
                H = torch.add(torch.matmul(H, W2), b)
                if l != self.num_Layers - 2:
                    H = torch.sin(H)
            else:
                W = self.weights2[l]
                b = self.biases2[l]
                H = torch.add(torch.matmul(H, W), b)
                H = torch.sin(H) 
        # Residual sum  
        Y = torch.sin(torch.add(H, X))
        return Y
    
    def set_device(self, device):
        # Weight, biases
        self.weights2 = [0 for i in range(self.num_Layers - 1)]
        self.biases2 = [0 for i in range(self.num_Layers - 1)]
        for l in range(0, self.num_Layers - 1):
            self.weights2[l] = self.weights[l].data
            self.weights2[l] = self.weights2[l].to(device)
            self.biases2[l] = self.biases[l].data
            self.biases2[l] = self.biases2[l].to(device)
        # Mask
        self.masks2 = [0 for i in range(self.num_Layers - 3)]
        for l in range(0, self.num_Layers - 3):
            self.masks2[l] = self.masks[l].data
            self.masks2[l] = self.masks2[l].to(device)

# BsPINN Neural Network
class BsPINN(nn.Module):
    def __init__(self, Layers, num_sample, num_boundary, boundary_weight, lb_X, ub_X, num_res, dim_x, dim_y, coe):
        super(BsPINN, self).__init__()
        # Initialize parameters
        self.lb_X = lb_X
        self.ub_X = ub_X
        self.Layers = Layers
        self.num_sample = num_sample
        self.boundary_weight = boundary_weight
        self.num_res = num_res
        self.params = []
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_res_fc = Layers[0]  # Width of fully connected layers in the residual block, all layers have the same width
        self.num_boundary = num_boundary
        self.coe = coe
        # Initialize the first fully connected layer
        self.fc_first = nn.Linear(dim_x, self.dim_res_fc)
        nn.init.xavier_uniform_(self.fc_first.weight, gain=1)
        nn.init.zeros_(self.fc_first.bias)
        self.params.append(self.fc_first.weight)
        self.params.append(self.fc_first.bias)
        # Initialize residual layers
        self.res_blocks = nn.ModuleList([Residual(self.Layers) for i in range(self.num_res)])
        for i in range(self.num_res):
            self.params.extend(self.res_blocks[i].params)
        # Initialize the last fully connected layer
        self.fc_last = nn.Linear(self.dim_res_fc, self.dim_y)
        nn.init.xavier_uniform_(self.fc_last.weight, gain=1)
        nn.init.zeros_(self.fc_last.bias)
        self.params.append(self.fc_last.weight)
        self.params.append(self.fc_last.bias)
        # Calculate the number of parameters
        self.num_params = sum(p.numel() for p in self.fc_first.parameters() if p.requires_grad)
        for n in range(self.num_res):
            self.num_params += self.res_blocks[i].num_params
        self.num_params += sum(p.numel() for p in self.fc_last.parameters() if p.requires_grad)
    
    # Fully connected neural network part
    def neural_net(self, X):
        # Data preprocessing, lb and ub used for training and testing should be the same, otherwise the neural network used for training and testing will be different.
        X = 2.0 * (X - self.lb_X) / (self.ub_X - self.lb_X) - 1.0
        H = X.float()
        # ResNet part
        H = self.fc_first(H)
        for i in range(self.num_res):
            H = self.res_blocks[i](H)
        Y = self.fc_last(H)

        return Y

    def neural_net2(self, X):
        # Data preprocessing, lb and ub used for training and testing should be the same, otherwise the neural network used for training and testing will be different.
        X = 2.0 * (X - self.lb_X2) / (self.ub_X2 - self.lb_X2) - 1.0
        H = X.float()
        # Neural network part
        H = torch.add(torch.matmul(H, self.wf2), self.bf2)
        for i in range(self.num_res): 
            H = self.res_blocks[i].forward2(H)
        Y = torch.add(torch.matmul(H, self.wl2), self.bl2)

        return Y

    # PDE part
    def he_net(self, X):
        # Equation
        X_e = [0 for i in range(self.dim_x)]
        for i in range(self.dim_x):
            X_e[i] = X[0:self.num_sample, i : i + 1].clone()
            X_e[i] = X_e[i].requires_grad_()
        u_e = self.neural_net(torch.cat(X_e, dim = 1))
        dudx = [grad(u_e, X_e[i]) for i in range(self.dim_x)]
        dudx2 = [grad(dudx[i], X_e[i]) for i in range(self.dim_x)]
        dudx2 = torch.cat(dudx2, dim=1)  # self.num_sample * dim_x
        Laplace_u = torch.sum(dudx2, dim=1, keepdim=True)  # self.num_sample * 1
        sum_x_e = torch.sum(X[0:self.num_sample, :], dim=1, keepdim=True)  # self.num_sample * 1
        f = self.dim_x * (self.coe ** 2) * torch.sin(self.coe * sum_x_e) - 2 / self.dim_x  # Right-hand side of the equation
        equation = - Laplace_u - f

        # Boundary conditions
        X_b = [0 for i in range(self.dim_x)]
        for i in range(self.dim_x):
            X_b[i] = X[self.num_sample:, i : i + 1].clone()
        u_b = self.neural_net(torch.cat(X_b, dim = 1))  # self.num_boundary * 1
        sum_x_b = torch.sum(X[self.num_sample:, :], dim=1, keepdim=True)  # self.num_boundary * 1
        mean_x_b = sum_x_b / self.dim_x
        Dvalue = torch.pow(mean_x_b, 2) + torch.sin(self.coe * sum_x_b)  # Dirichlet boundary value
        boundary = u_b - Dvalue

        # Total displacement
        u = torch.cat([u_e, u_b], dim = 0)  # (self.num_sample + self.num_boundary) * 1

        return u, equation, boundary

    # Loss function
    def loss(self, X_train):
        # Calculate equation and boundary condition terms
        _, equation, boundary = self.he_net(X_train)
        
        # Calculate total error
        loss_e = torch.mean(torch.square(equation))
        loss_b = torch.mean(torch.square(boundary))
        loss_all = loss_e + self.boundary_weight * loss_b

        return loss_all
    
    # Predict the value of u at point X
    def predict(self, X):
        u_pred = self.neural_net(X)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred 
    
    def predict2(self, X):
        u_pred = self.neural_net2(X)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred 
    
    # Parameters should be assigned to a new variable using .data, then use .to(device)
    def set_device(self, device): 
        # Normalization variables
        self.lb_X2 = self.lb_X.data
        self.lb_X2 = self.lb_X2.to(device)
        self.ub_X2 = self.ub_X.data
        self.ub_X2 = self.ub_X2.to(device)
        # First fully connected layer
        self.wf2 = self.fc_first.weight.data.T
        self.wf2 = self.wf2.to(device)
        self.bf2 = self.fc_first.bias.data
        self.bf2 = self.bf2.to(device)
        # Residual blocks
        for i in range(self.num_res):
            self.res_blocks[i].set_device(device)
        # Last fully connected layer
        self.wl2 = self.fc_last.weight.data.T
        self.wl2 = self.wl2.to(device)
        self.bl2 = self.fc_last.bias.data
        self.bl2 = self.bl2.to(device)
        
    # Calculate relative error: takes about 5s each time
    def rel_error(self):
        # Data conversion 1
        self.set_device(device_cpu)
        
        # Calculate error
        u_pred = self.predict2(X_pred).flatten()
        u_truth = solution(points)
        u_L2RE = np.sum(weights * np.square(u_pred - u_truth)) / np.sum((weights * np.square(u_truth))) 

        return u_L2RE

# main function
if __name__ == "__main__":
    # Gauss data
    dim_gauss = 4
    points1 = np.loadtxt(path + "../data/gauss_points_%d_1.txt" % dim_gauss, dtype = float, delimiter=' ')
    points2 = np.loadtxt(path + "../data/gauss_points_%d_2.txt" % dim_gauss, dtype = float, delimiter=' ')
    points = np.r_[points1, points2]
    weights = np.loadtxt(path + "../data/gauss_weights_%d.txt" % dim_gauss, dtype = float, delimiter=' ')
    X_pred = torch.from_numpy(points).float().to(device_cpu)
    
    # seed
    seeds = [11]
    csv_data = []
    for seed in seeds: 
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Parameter settings
        ## Equation-related
        lb_geom = -1 # Lower bound of x
        ub_geom = 1 # Upper bound of x
        dim_x = 10 # Dimension of x
        dim_y = 1 # Dimension of y
        coe = 0.6 * np.pi # Coefficient in sin
        
        ## Neural network-related
        train = True
        name = "BsPINN_256-32_Adam_v1"
        epochs = 10000 # Number of optimizer iterations
        num_sample = 4000
        num_boundary = 2000 # Must be divisible by dim_x
        Layers = [256, 128, 64, 32] # Fully connected network structure in each residual block, the number of neurons in each layer must be equal
        learning_rate = 0.001 # Initial learning rate
        boundary_weight = 1 # Setting this too high may cause gradient explosion
        num_res = 2 # Number of residual blocks
        h = int(1000) # Learning rate decay related
        r = 0.95 # Multiply learning rate every h iterations
        weight_decay = 0.001 # Coefficient of L2 regularization term to prevent sudden increase in loss
        
        ## Plot-related
        align = False
        lb_loss = 1e-3
        ub_loss = 1e21
        lb_error = 1e-2
        ub_error = 1e1
        lb_u = -1.1
        ub_u = 1.1
        lb_diff = -0.05
        ub_diff = 0.05
        error_param = 5 # Draw (20 * error_param) error points, each drawing takes about 6s. epochs needs to be divisible by (20 * error_param).
        record_error = True
        
        # Auxiliary variables
        name = name + ("_%d" % epochs)
        print("\n\n***** name = %s *****" % name)
        print("seed = %d" % seed)
        output_path = path + ('./output_BsPINN/')
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('./output_BsPINN/%s/' % name)
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('./output_BsPINN/%s/train_%d/' % (name, seed))
        if not os.path.exists(output_path): os.mkdir(output_path)
        
        # Gaussian integral data (do not set as self. variable, otherwise torch.save will be very slow)
        X_pred = torch.from_numpy(points).float().to(device_cpu)

        # Generate dataset
        if train:
            print("Loading data")
            ## Training data
            X_train = load_data(lb_geom, ub_geom, num_sample, num_boundary, dim_x)
            lb_X = X_train.min(0) # Get the maximum value of each dimension of the training set
            ub_X = X_train.max(0) # Get the minimum value of each dimension of the training set
            lb_X = torch.from_numpy(lb_X).float().to(device)
            ub_X = torch.from_numpy(ub_X).float().to(device)
            X_train = torch.from_numpy(X_train).float().to(device)
            
            # Declare the neural network instance
            model = BsPINN(Layers, num_sample, num_boundary, boundary_weight, lb_X, ub_X, num_res, dim_x, dim_y, coe)
            model = nn.DataParallel(model)
            model = model.module
            model.to(device)
            print(model) # Print network summary
            params = model.num_params
            print("params = %d" % params)
            
            # Optimizer
            optimizer = optim.Adam(model.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)

            # Training
            start = time.time()
            ## Adam
            loss_list = []
            error_list = [] # Save the average relative error value on the test set
            min_loss = 9999999
            print("Start training!")
            for it in range(epochs):
                # Optimizer training
                Loss = model.loss(X_train)
                optimizer.zero_grad(set_to_none=True)
                Loss.backward() 
                optimizer.step()

                # Decay learning rate
                if (it + 1) % h == 0: # Exponential decay
                    learning_rate *= r
                    reset_lr(optimizer, learning_rate)

                # Save the loss value and relative error, and save the model with the smallest training loss
                loss_val = Loss.cpu().detach().numpy() 
                loss_list.append(loss_val)
                
                # Save the model with the minimum train loss
                if loss_val < min_loss: # This step makes BsPINN slower than PINN
                    torch.save(model, output_path + 'network.pkl') 
                    min_loss = loss_val
                        
                # Save error curve
                if record_error and (it + 1) % (epochs/20/error_param) == 0:
                    u_L2RE = model.rel_error() # Takes 6s each time
                    error_list.append(u_L2RE)

                # Output
                if (it + 1) % (epochs/20) == 0:
                    if record_error:
                        print("It = %d, loss = %.8f, u_L2RE = %.8f, finish: %d%%" % ((it + 1), loss_val, u_L2RE, (it + 1) / epochs * 100))
                    else:
                        print("It = %d, loss = %.8f, finish: %d%%" % ((it + 1), loss_val, (it + 1) / epochs * 100))
                
            ## Post-processing
            end = time.time()
            train_time = end - start
            loss_list = np.array(loss_list).flatten()
            error_list = np.array(error_list).flatten()
            min_loss = np.min(loss_list)
            np.savetxt(output_path + "loss.txt", loss_list, fmt="%s",delimiter=' ')
            if record_error:
                np.savetxt(output_path + "error.txt", error_list, fmt="%s",delimiter=' ')
            print("time = %.2fs" % train_time)
            print("min_loss = %.8f" % min_loss)

        # Save loss curve
        plt.rcParams['font.size'] = 20
        loss_list = np.loadtxt(output_path + "loss.txt", dtype = float, delimiter=' ')
        plt.figure()
        plt.semilogy(loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if align:
            plt.ylim(lb_loss, ub_loss) # Consistent scale with BiPINN
            plt.savefig(output_path + 'loss_aligned.pdf', format="pdf", dpi=100, bbox_inches="tight")
        else:
            plt.savefig(output_path + 'loss.pdf', format="pdf", dpi=100, bbox_inches="tight")
            
        # Save error curve
        if record_error:
            fig, ax = plt.subplots()
            error_list = np.loadtxt(output_path + "error.txt", dtype = float, delimiter=' ')
            plt.semilogy(error_list)
            plt.xlabel('Epoch')
            plt.ylabel('Relative error')
            tool = [0, 4, 8, 12, 16, 20]
            plt.xticks([tool[i] * error_param for i in range(len(tool))], [str(0), str(int(epochs * 0.2)), str(int(epochs * 0.4)), str(int(epochs * 0.6)), str(int(epochs * 0.8)), str(int(epochs))])
            if align:
                plt.ylim(lb_error, ub_error) # Consistent scale with BiPINN
                plt.savefig(output_path + 'error_aligned.pdf', format="pdf", dpi=300, bbox_inches="tight")
            else:
                plt.savefig(output_path + 'error.pdf', format="pdf", dpi=300, bbox_inches="tight")

        # Calculate the relative error of u on grid points
        # dim_gauss = 4
        X_pred = torch.from_numpy(points).float().to(device_cpu)
        model2 = torch.load(output_path + 'network.pkl', map_location=device_cpu) 
        u_pred = model2.predict(X_pred).flatten()
        u_truth = solution(points)
        u_L2RE = np.sum(weights * np.square(u_pred - u_truth)) / np.sum((weights * np.square(u_truth)))
        print("u_L2RE = %.10f when using Gauss L2RE" % u_L2RE)
        np.savetxt(output_path + "u_truth.txt", u_truth, fmt="%s", delimiter=' ')
        np.savetxt(output_path + "u_pred_bipinn.txt", u_pred, fmt="%s", delimiter=' ')
        
        # Prepare data for drawing
        dim = 500
        X1 = np.linspace(lb_geom, ub_geom, dim)
        X2 = np.linspace(lb_geom, ub_geom, dim)
        X1, X2 = np.meshgrid(X1, X2)
        X1 = X1.flatten().reshape(dim*dim,1)
        X2 = X2.flatten().reshape(dim*dim,1)
        points2 = np.c_[X1, X2] # N * 2
        tool = np.zeros((X1.shape[0], dim_x - 2)) # The last 8 elements should all be 1
        points2 = np.c_[points2, tool] # N * dim_x, N test points.
        u_truth = solution(points2)
        X_pred2 = points2
        X_pred2 = torch.from_numpy(X_pred2).float().to(device_cpu)
        model2 = torch.load(output_path + 'network.pkl', map_location=device_cpu) 
        if train: model2.set_device(device_cpu)
        u_pred = model2.predict(X_pred2).flatten()
        u_L2RE2 = np.linalg.norm((u_truth - u_pred),ord=2) / (np.linalg.norm(u_truth, ord=2))
        # print("u_L2RE = %.8f when drawing." % u_L2RE2)

        # Draw predicted solution image
        fig, ax = plt.subplots()
        if align:
                levels = np.arange(lb_u, ub_u + 1e-8, (ub_u - lb_u) / 100)
        else:
            levels = np.arange(min(u_pred) - abs(max(u_pred) - min(u_pred)) / 10, max(u_pred) + abs(max(u_pred) - min(u_pred)) / 10, (max(u_pred) - min(u_pred)) / 100) 
        cs = ax.contourf(X1.reshape(dim,dim), X2.reshape(dim,dim), u_pred.reshape(dim,dim), levels,cmap='jet')
        cbar = fig.colorbar(cs)
        plt.xticks([-1,-0.5,0,0.5,1])
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('$u$(BsPINN)')
        if align:
            plt.savefig(output_path + "highpos_bipinn.png", format="png", dpi=100, bbox_inches="tight")
        else:
            plt.savefig(output_path + "u_pred.png", format="png", dpi=100, bbox_inches="tight")
            
        # Draw accurate solution image
        plt.rcParams['font.size'] = 30
        fig, ax = plt.subplots()
        if align:
                levels = np.arange(lb_u, ub_u + 1e-8, (ub_u - lb_u) / 100)
        else:
            levels = np.arange(min(u_truth) - abs(max(u_truth) - min(u_truth)) / 10, max(u_truth) + abs(max(u_truth) - min(u_truth)) / 10, (max(u_truth) - min(u_truth)) / 100) 
        cs = ax.contourf(X1.reshape(dim,dim), X2.reshape(dim,dim), u_truth.reshape(dim,dim), levels,cmap='jet')
        cbar = fig.colorbar(cs)
        plt.xticks([-1,-0.5,0,0.5,1])
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('$u$(truth)')
        if align:
            plt.savefig(output_path + "highpos_truth.png", format="png", dpi=100, bbox_inches="tight")
        else:
            plt.savefig(output_path + "u_truth.png", format="png", dpi=100, bbox_inches="tight")
            
        # Draw error image
        u_diff = u_truth - u_pred
        plt.rcParams['font.size'] = 30
        fig, ax = plt.subplots()
        if align:
                levels = np.arange(lb_diff, ub_diff + 1e-8, (ub_diff - lb_diff) / 100)
        else:
            levels = np.arange(min(u_diff) - abs(max(u_diff) - min(u_diff)) / 10, max(u_diff) + abs(max(u_diff) - min(u_diff)) / 10, (max(u_diff) - min(u_diff)) / 100) 
        cs = ax.contourf(X1.reshape(dim,dim), X2.reshape(dim,dim), u_diff.reshape(dim,dim), levels,cmap='jet')
        cbar = fig.colorbar(cs)
        plt.xticks([-1,-0.5,0,0.5,1])
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('$u$(difference)')
        if align:
            plt.savefig(output_path + "u_diff_aligned.png", format="png", dpi=100, bbox_inches="tight")
        else:
            plt.savefig(output_path + "u_diff.png", format="png", dpi=100, bbox_inches="tight")
            
        # Save error value
        csv_data.append([seed, u_L2RE])
    
    # Save as csv file
    output_path = path + ('./output_BsPINN/%s/' % name)
    file_write = open(output_path + 'record.csv','w')
    writer = csv.writer(file_write)
    writer.writerow(['seed','L2RE'])
    writer.writerows(csv_data)
    file_write.close()
