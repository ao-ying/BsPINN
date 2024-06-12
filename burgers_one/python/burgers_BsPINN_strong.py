import os
import csv
import numpy as np
from matplotlib import pyplot as plt
import time
import torch
import torch.nn as nn         
import torch.optim as optim             
from torch.nn.parameter import Parameter

# System Settings
torch.set_default_dtype(torch.float)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Device configuration
device_cpu = torch.device('cpu')
path = os.path.dirname(__file__) + "/"
torch.backends.cudnn.benchmark = True
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 30

# Compute partial derivatives
def grad(f,x):
    ret = torch.autograd.grad(f, x, torch.ones_like(f).to(device), retain_graph=True, create_graph=True)[0]
    return ret

# Output model parameter information
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.params)
    trainable_num = sum(p.numel() for p in net.params if p.requires_grad)
    return trainable_num
    
# Set the learning rate of optimizer to lr
def reset_lr(optimizer, lr):
    for params in optimizer.param_groups: 
            params['lr'] = lr

# Import datasets: interior domain, initial conditions, boundary conditions
def load_data(Nf, lb_x, ub_x, lb_t, ub_t, dimx, dimt):
    # Training set
    X_train = np.random.uniform([lb_x, lb_t], [ub_x, ub_t], (Nf, 2))
    
    # Test set
    ## Test points
    X = np.linspace(-1, 1, dimx) # X and T are consistent with the exact solution calculation program.
    T = np.linspace(0, 0.99, dimt)
    X, T = np.meshgrid(X, T)
    X = X.flatten().reshape(dimx * dimt, 1)
    T = T.flatten().reshape(dimx * dimt, 1)
    X_test = np.c_[X, T] # N * 2
    ## Test labels
    file_read = open(path + '../data/burgers_solution_u_%d_%d.csv' % (dimx, dimt),'r')
    reader = csv.reader(file_read)
    Y_test = np.array(list(reader)).astype(np.float64).T.flatten()
    
    return X_train, X_test, Y_test


# BsPINN Neural Network
class BsPINN(nn.Module):
    def __init__(self, Layers, Nf, lb_X, ub_X):
        super(BsPINN, self).__init__()
        # Initialize parameters
        self.iter = 0
        self.lb_X = lb_X
        self.ub_X = ub_X
        self.Layers = Layers
        self.Nf = Nf
        self.width = [Layers[0]] + [int(pow(2, i - 1) * Layers[i]) for i in range(1, len(Layers) - 1)] + [Layers[-1]]
        self.num_Layers = len(self.Layers)
        self.masks = self.construct_mask()
        self.num_param = self.cal_param()
        self.params = []
        self.act = torch.tanh
        # Initialize binary neural network
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
        # Middle hidden layers
        for l in range(1, self.num_Layers - 2):
            # Weight w
            tempw = torch.zeros(self.width[l], self.width[l + 1]).to(device)
            # Block diagonal matrix initialization
            for i in range(int(pow(2, l))):  # Traverse each small matrix and initialize
                tempw2 = torch.zeros(Layers[l], Layers[l + 1])
                w2 = Parameter(tempw2, requires_grad=True)
                nn.init.xavier_uniform_(w2, gain=1)
                row_index = int(i / 2)
                tempw[row_index * Layers[l]: (row_index + 1) * Layers[l], i * Layers[l + 1]: (i + 1) * Layers[l + 1]] = w2.data
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

    # Binary-connected neural network part
    def neural_net(self, X):
        # Extract coordinates x and t. Note that x and t must be extracted before normalization, not after.
        x = X[:, 0:1]
        t = X[:, 1:2]

        # Normalize
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

        # Force initial and boundary conditions
        sd = 0.1
        U = - torch.sin(np.pi * x) + torch.tanh((x + 1) / sd) * torch.tanh((1 - x) / sd) * torch.tanh(t / sd) * X  # Here X represents the output of the neural network, and U represents the final displacement value.

        return U

    def neural_net_channel(self, X, channel):
        # Data preprocessing
        X = 2.0 * (X - self.lb_X) / (self.ub_X - self.lb_X) - 1.0
        H = X.float()

        # First hidden layer
        num_Layers = len(self.Layers)
        l_out = torch.sin(torch.add(torch.matmul(H, self.weights[0][0]), self.biases[0][0]))
        temp = [[] for i in range(num_Layers - 2)]  # temp saves the output results of each network layer.
        temp[0].append(l_out)
        # Subsequent hidden layers
        for l in range(1, num_Layers - 2):
            for i in range(int(pow(2, l))):
                W = self.weights[l][i]
                b = self.biases[l][i]
                l_out = torch.sin(torch.add(torch.matmul(temp[l - 1][int(i / 2)], W), b))
                temp[l].append(l_out)
        # Last fully connected layer
        out = torch.zeros(X.shape[0], self.Layers[1]).to(device)
        out[:, self.Layers[-2] * channel: self.Layers[-2] * (channel + 1)] = temp[num_Layers - 3][channel]
        Y = torch.add(torch.matmul(out, self.w_last), self.b_last)
        return Y

    # PDE part
    def he_net(self, X):
        # Equation
        x_e = X[0: self.Nf, 0:1].clone()
        x_e = x_e.requires_grad_()
        t_e = X[0: self.Nf, 1:2].clone()
        t_e = t_e.requires_grad_()
        u_e = self.neural_net(torch.cat([x_e, t_e], dim=1))
        u_t = grad(u_e, t_e)
        u_x = grad(u_e, x_e)
        u_xx = grad(u_x, x_e)
        equation = u_t + u_e * u_x - (0.01 / np.pi) * u_xx

        return equation

    # Loss function
    def loss(self, X_train):
        # Calculate the equation and boundary condition terms
        equation = self.he_net(X_train)

        # Calculate the total error
        loss = torch.mean(torch.square(equation))

        return loss

    # Predict the value of u at the point corresponding to X
    def predict(self, X):
        u_pred = self.neural_net(X)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred

    def data_transfer(self, min_loss):
        self.min_loss = min_loss

    # Predict the image of a certain channel
    def predict_channel(self, X, channel):
        u_pred = self.neural_net_channel(X, channel)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred

    # Calculate relative error
    def rel_error(self):
        # The current model of a certain training part should be used here, not the current optimal model.
        u_pred = self.neural_net(X_test).cpu().detach().numpy().flatten()  # Before calculating the error, flatten() must be used
        u_truth = Y_test.cpu().detach().numpy()
        u_L2RE = np.linalg.norm((u_truth - u_pred), ord=2) / (np.linalg.norm(u_truth, ord=2))
        u_L1 = np.linalg.norm((u_truth - u_pred), ord=1) / u_truth.shape[0]
        return u_L2RE, u_L1

    
# main function
if __name__ == "__main__":
    seeds = [27]
    csv_data = []
    for seed in seeds: 
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Parameter settings
        ## Equation related
        lb_x = -1 # Lower bound of x
        ub_x = 1 # Upper bound of x
        lb_t = 0
        ub_t = 1
        dimx = 256 # Number of points in the x and t directions of the test set.
        dimt = 100
        
        ## Neural network related
        train = True
        name = "128-8_v22" 
        epochs = 200000 # Number of iterations of the adam optimizer
        Nf = 100000 # Number of training points in the solution domain
        Layers = [2,128,64,32,16,8,1]
        learning_rate = 0.001 # Initial learning rate 
        patience = max(10, epochs / 10) # Patience number for plateau learning rate decay
        weight_decay = 0 # L2 regularization coefficient
        
        ## Plotting related
        align = False
        lb_loss = 1e-3
        ub_loss = 1e21
        lb_u = -1.2
        ub_u = 1.2
        lb_error = 0.2
        ub_error = 1.1
        lb_diff = -0.00035
        ub_diff = 0.00035
        error_param = 5 # Draw (20 * error_param) error points, epochs need to be divisible by (20 * error_param).
        
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

        if train:
            # Generate dataset
            print("Loading data")
            X_train, X_test, Y_test = load_data(Nf, lb_x, ub_x, lb_t, ub_t, dimx, dimt)
            lb_X = X_train.min(0) # Get the maximum value of each dimension of the training set
            ub_X = X_train.max(0) # Get the minimum value of each dimension of the training set
            lb_X = torch.from_numpy(lb_X).float().to(device)
            ub_X = torch.from_numpy(ub_X).float().to(device)
            X_train = torch.from_numpy(X_train).float().to(device)
            X_test = torch.from_numpy(X_test).float().to(device)
            Y_test = torch.from_numpy(Y_test).float().to(device)

            # Declare neural network instance
            model = BsPINN(Layers, Nf, lb_X, ub_X)
            model = nn.DataParallel(model)
            model = model.module
            model.to(device)
            print(model) # Print network summary
            params = model.num_param # Output parameter information
            print("params = %d" % params)
            
            # Adam optimizer
            optimizer = optim.Adam(model.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, verbose=True, min_lr=1e-6) # Plateau learning rate decay

            # Training
            start = time.time()
            ## adam
            loss_list = []
            error_list = []
            min_loss = 9999999
            print("Start training!")
            for it in range(epochs):
                # Optimizer training
                Loss = model.loss(X_train)
                optimizer.zero_grad(set_to_none=True)
                Loss.backward() 
                optimizer.step()

                # Learning rate decay
                scheduler.step(Loss)

                # Save loss values and relative errors, and save the model with the minimum training error
                loss_val = Loss.cpu().detach().numpy() # If running with GPU, data must be transferred to CPU first
                loss_list.append(loss_val)
                
                # save the model with the minimum train loss
                if loss_val < min_loss: 
                    torch.save(model, output_path + 'network.pkl') 
                    min_loss = loss_val
                
                # Save error curves
                if (it + 1) % (epochs/20/error_param) == 0:
                    u_L2RE, u_L1 = model.rel_error()
                    error_list.append(u_L2RE)

                # Output
                if (it + 1) % (epochs/20) == 0:
                    print("It = %d, loss = %.8f, u_L2RE = %.8f, u_L1 = %.8f, finish: %d%%" % ((it + 1), loss_val, u_L2RE, u_L1, (it + 1) / epochs * 100))
                
            # Subsequent processing
            end = time.time()
            train_time = end - start
            loss_list = np.array(loss_list).flatten()
            np.savetxt(output_path + "loss.txt", loss_list, fmt="%s",delimiter=' ')
            np.savetxt(output_path + "error.txt", error_list, fmt="%s",delimiter=' ')
            min_loss = np.min(loss_list)
            final_lr = optimizer.param_groups[0]['lr']
            print("time = %.2fs" % train_time)
            print("min_loss = %.8f" % min_loss)
        else:
            X_train, X_test, Y_test = load_data(Nf, lb_x, ub_x, lb_t, ub_t, dimx, dimt)
            X_test = torch.from_numpy(X_test).float().to(device)
            Y_test = torch.from_numpy(Y_test).float().to(device)
        torch.cuda.empty_cache() # Release GPU memory

        # Save loss curve
        plt.rcParams['font.size'] = 20
        loss_list = np.loadtxt(output_path + "loss.txt", dtype = float, delimiter=' ')
        plt.figure()
        plt.semilogy(loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if align:
            plt.ylim(lb_loss, ub_loss) # Align with BiPINN
            plt.savefig(output_path + 'loss_aligned.pdf', format="pdf", dpi=150, bbox_inches="tight")
        else:
            plt.savefig(output_path + 'loss.pdf', format="pdf", dpi=150, bbox_inches="tight")
            
        # Save error curve
        fig, ax = plt.subplots()
        error_list = np.loadtxt(output_path + "error.txt", dtype = float, delimiter=' ')
        plt.semilogy(error_list)
        plt.xlabel('Epoch')
        plt.ylabel('Relative error')
        tool = [0, 4, 8, 12, 16, 20]
        plt.xticks([tool[i] * error_param for i in range(len(tool))], [str(0), str(int(epochs * 0.2)), str(int(epochs * 0.4)), str(int(epochs * 0.6)), str(int(epochs * 0.8)), str(int(epochs))])
        if align:
            plt.ylim(lb_error, ub_error) # Align with BiPINN
            plt.savefig(output_path + 'error_aligned.pdf', format="pdf", dpi=150, bbox_inches="tight")
        else:
            plt.savefig(output_path + 'error.pdf', format="pdf", dpi=150, bbox_inches="tight")

        # Calculate relative error of u on grid points
        X = np.linspace(-1, 1, dimx)
        T = np.linspace(0, 0.99, dimt)
        X, T = np.meshgrid(X, T)
        X = X.flatten().reshape(dimx * dimt, 1)
        T = T.flatten().reshape(dimx * dimt, 1)
        X_pred = np.c_[X, T] # N * 2
        X_pred = torch.from_numpy(X_pred).float().to(device)
        model2 = torch.load(output_path + 'network.pkl', map_location=device) 
        u_pred = model2.predict(X_test).flatten()
        u_truth = Y_test.cpu().detach().numpy()
        u_L2RE = np.linalg.norm((u_truth - u_pred), ord=2) / (np.linalg.norm(u_truth, ord=2))
        print("u_L2RE = %.8f" % u_L2RE)
        u_L1 = np.linalg.norm((u_truth - u_pred), ord=1) / u_truth.shape[0]
        print("u_L1 = %.8f" % u_L1)

        # Plot predicted solution
        plt.rcParams['font.size'] = 30
        fig, ax = plt.subplots()
        if align:
            levels = np.arange(lb_u, ub_u + 1e-8, (ub_u - lb_u) / 100)
        else:
            levels = np.arange(min(u_pred) - abs(max(u_pred) - min(u_pred)) / 10, max(u_pred) + abs(max(u_pred) - min(u_pred)) / 10, (max(u_pred) - min(u_pred)) / 100) 
        cs = ax.contourf(X.reshape(dimt, dimx), T.reshape(dimt, dimx), u_pred.reshape(dimt, dimx), levels, cmap='jet')
        cbar = fig.colorbar(cs)
        plt.xticks([-1, -0.5, 0, 0.5, 1])
        plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])
        plt.xlabel('$x$')
        plt.ylabel('$t$')
        plt.title('$u$(BsPINN)')
        if align:
            plt.savefig(output_path + "u_pred_aligned.png", format="png", dpi=150, bbox_inches="tight")
        else:
            plt.savefig(output_path + "u_pred.png", format="png", dpi=150, bbox_inches="tight")
            
        # Plot exact solution
        fig, ax = plt.subplots()
        if align:
            levels = np.arange(lb_u, ub_u + 1e-8, (ub_u - lb_u) / 100)
        else:
            levels = np.arange(min(u_truth) - abs(max(u_truth) - min(u_truth)) / 10, max(u_truth) + abs(max(u_truth) - min(u_truth)) / 10, (max(u_truth) - min(u_truth)) / 100) 
        cs = ax.contourf(X.reshape(dimt, dimx), T.reshape(dimt, dimx), u_truth.reshape(dimt, dimx), levels, cmap='jet')
        cbar = fig.colorbar(cs)
        plt.xticks([-1, -0.5, 0, 0.5, 1])
        plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])
        plt.xlabel('$x$')
        plt.ylabel('$t$')
        plt.title('$u$(truth)')
        if align:
            plt.savefig(output_path + "u_truth_aligned.png", format="png", dpi=150, bbox_inches="tight")
        else:
            plt.savefig(output_path + "u_truth.png", format="png", dpi=150, bbox_inches="tight")
            
        # Plot error image
        u_diff = u_truth - u_pred
        u_diff_inf = np.max(np.abs(u_diff))
        print("The infinity norm of diff is %.8f" % u_diff_inf)
        fig, ax = plt.subplots()
        if align:
            levels = np.arange(lb_diff, ub_diff + 1e-8, (ub_diff - lb_diff) / 100)
        else:
            levels = np.arange(min(u_diff) - abs(max(u_diff) - min(u_diff)) / 10, max(u_diff) + abs(max(u_diff) - min(u_diff)) / 10, (max(u_diff) - min(u_diff)) / 100) 
        cs = ax.contourf(X.reshape(dimt, dimx), T.reshape(dimt, dimx), u_diff.reshape(dimt, dimx), levels, cmap='jet')
        cbar = fig.colorbar(cs)
        plt.xticks([-1, -0.5, 0, 0.5, 1])
        plt.yticks([0.00, 0.25, 0.50, 0.75, 1.00])
        plt.xlabel('$x$')
        plt.ylabel('$t$')
        plt.title('Point-wise Error(BsPINN)', fontsize=20)
        if align:
            plt.savefig(output_path + "burgers_diff_bspinn3.png", format="png", dpi=150, bbox_inches="tight")
        else:
            plt.savefig(output_path + "u_diff.png", format="png", dpi=150, bbox_inches="tight")

        # Save error values
        csv_data.append([seed, u_L2RE, u_L1])

    # Save as csv file
    output_path = path + ('./output_BsPINN/%s/' % name)
    file_write = open(output_path + 'record.csv','w')
    writer = csv.writer(file_write)
    writer.writerow(['seed','L2RE','L1'])
    writer.writerows(csv_data)
    file_write.close()

        
        