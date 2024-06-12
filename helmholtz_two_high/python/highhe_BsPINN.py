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
def solution(x,y):
    return np.sin(kappa * x) * np.sin(kappa * y)

# Compute gradient
def grad(f,x):
    ret = torch.autograd.grad(f, x, torch.ones_like(f).to(device), retain_graph=True, create_graph=True)[0]
    return ret

# Output model parameter information
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.params)
    trainable_num = sum(p.numel() for p in net.params if p.requires_grad)
    return trainable_num

# Generate dataset
# num_domain: number of training data points, num_boundary: four times of this is the number of boundary sampling points, dim_test: number of sampling points on one edge of the test set.
def load_data(num_domain, num_boundary, dim_test, lb_geom, ub_geom):
    # Generate training sampling points
    delta = 0.01
    X_train = np.random.uniform([lb_geom + delta, lb_geom + delta], [ub_geom - delta, ub_geom - delta], (num_domain, 2))
    # Boundary
    num_boundary_per_edge = int(num_boundary / 4)
    points = np.linspace([lb_geom,lb_geom],[ub_geom,lb_geom],num_boundary_per_edge,endpoint=False) # Bottom
    X_train = np.r_[X_train, points]
    points = np.linspace([ub_geom,lb_geom],[ub_geom,ub_geom],num_boundary_per_edge,endpoint=False) # Right
    X_train = np.r_[X_train, points]
    points = np.linspace([lb_geom,lb_geom],[lb_geom,ub_geom],num_boundary_per_edge,endpoint=False) # Left
    X_train = np.r_[X_train, points]
    points = np.linspace([lb_geom,ub_geom],[ub_geom,ub_geom],num_boundary_per_edge,endpoint=False) # Top
    X_train = np.r_[X_train, points]

    # Test set
    X = np.linspace(lb_geom, ub_geom, dim_test)
    Y = np.linspace(lb_geom, ub_geom, dim_test)
    X, Y = np.meshgrid(X, Y)
    X = X.flatten().reshape(dim_test*dim_test,1)
    Y = Y.flatten().reshape(dim_test*dim_test,1)
    X_test = np.c_[X,Y]
    Y_test = solution(X.flatten(),Y.flatten())
    Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))

    return X_train, X_test, Y_test


class BsPINN(nn.Module):
    def __init__(self, Layers, k0, num_domain, boundary_weight, lb_X, ub_X):
        super(BsPINN, self).__init__()
        # Initialize parameters
        self.k0 = k0
        self.lb_X = lb_X
        self.ub_X = ub_X
        self.Layers = Layers
        self.num_Layers = len(self.Layers)
        self.num_domain = num_domain
        self.boundary_weight = boundary_weight
        self.params = []
        self.act = torch.sin
        # binary neural network related
        self.width = [Layers[0]] + [int(pow(2, i - 1) * Layers[i]) for i in range(1,len(Layers) - 1)] + [Layers[-1]] 
        self.masks = self.construct_mask()
        self.num_param = self.cal_param()
        self.weights, self.biases =  self.initialize_NN(self.Layers)
    
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
        for l in range(1,self.num_Layers - 2):
            # Weight w
            tempw = torch.zeros(self.width[l], self.width[l+1]).to(device)
            # Initialize block diagonal matrix
            for i in range(int(pow(2,l))): 
                tempw2 = torch.zeros(Layers[l], Layers[l+1])
                w2 = Parameter(tempw2, requires_grad=True)
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
            blocks = [torch.ones((blocksize1,blocksize2)) for i in range(num_blocks)]
            mask = torch.block_diag(*blocks).to(device)
            masks.append(mask)
        return masks 
    
    # BsNN part
    def neural_net(self, X):
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
    def he_net(self, x, y):
        # Equation
        x_e = x[0:self.num_domain].clone()
        x_e = x_e.requires_grad_()
        y_e = y[0:self.num_domain].clone()
        y_e = y_e.requires_grad_()
        u_e = self.neural_net(torch.cat([x_e,y_e], dim = 1))
        f = self.k0 ** 2 * torch.sin(self.k0 * x_e) * torch.sin(self.k0 * y_e)
        u_x = grad(u_e, x_e)
        u_xx = grad(u_x, x_e)
        u_y = grad(u_e, y_e)
        u_yy = grad(u_y, y_e)
        equation = - u_xx - u_yy - self.k0 ** 2 * u_e - f

        # Boundary conditions
        x_b = x[self.num_domain:].clone()
        y_b = y[self.num_domain:].clone()
        u_b = self.neural_net(torch.cat([x_b, y_b], dim = 1))
        boundary = u_b - 0

        return equation, boundary

    # Loss function
    def loss(self,X_train):
        # Calculate equation and boundary condition terms
        x = X_train[:,0:1]
        y = X_train[:,1:2]
        equation, boundary = self.he_net(x, y)
        
        # Calculate total error
        loss_e = torch.mean(torch.square(equation))
        loss_b = torch.mean(torch.square(boundary))
        loss_all = loss_e + self.boundary_weight * loss_b

        return loss_all
    
    # test function used in training
    def test(self):
        u_pred = self.neural_net(X_test)
        u_mae = torch.mean(torch.abs(Y_test - u_pred)).cpu().detach().numpy()
        return u_mae
    
    # Calculate relative error
    def rel_error(self):
        u_pred = self.neural_net(X_test).cpu().detach().numpy()
        u_truth = Y_test.cpu().detach().numpy()
        u_error = np.linalg.norm((u_truth - u_pred), ord=2) / (np.linalg.norm(u_truth, ord=2))
        return u_error
    
    # Predict the value of u at point X
    def predict(self, X):
        u_pred = self.neural_net(X)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred

# main function
if __name__ == "__main__":
    seeds = [17]
    csv_data = []
    for seed in seeds:
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Parameter settings
        ## Neural network related
        train = True
        name = "512-32_v16"
        epochs = 50000 # Number of iterations for adam optimizer
        num_domain = 60000 # Number of training points within the solution domain
        num_boundary = 5000 # Number of boundary condition training points
        dim_test = 500 
        Layers = [2, 512, 256, 128, 64, 32, 1]
        learning_rate = 0.01 
        shuffle = False
        boundary_weight = 1
        patience = 2500 
        weight_decay = 0
        
        ## Equation related
        lb_geom = 0
        ub_geom = 1
        kappa = 24 * np.pi # kappa value
        
        ## Plot related
        align = False 
        lb_loss = 4e2
        ub_loss = 1e9
        lb_u = -1.85
        ub_u = 1.85
        lb_error = 0.2
        ub_error = 1.1
        error_param = 5 # Plot (20 * error_param) error points, epochs should be divisible by (20 * error_param).
        record_time = True
        
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
        if record_time:
            time_messsage = []

        if train:
            # Generate dataset
            print("Loading data.")
            X_train, X_test, Y_test = load_data(num_domain, num_boundary, dim_test, lb_geom, ub_geom)
            lb_X = X_train.min(0)
            ub_X = X_train.max(0)
            lb_X = torch.from_numpy(lb_X).float().to(device)
            ub_X = torch.from_numpy(ub_X).float().to(device)
            X_train = torch.from_numpy(X_train).float().to(device)
            X_test = torch.from_numpy(X_test).float().to(device)
            Y_test = torch.from_numpy(Y_test).float().to(device)

            # Declare neural network instance
            model = BsPINN(Layers, kappa, num_domain, boundary_weight, lb_X, ub_X)
            model = nn.DataParallel(model)
            model = model.module
            model.to(device)
            print(model) # Print network summary
            params = model.num_param # Output parameter information
            print("Total number of parameters: %d" % params)
            
            # Adam optimizer
            optimizer = optim.Adam(model.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, verbose=True, min_lr=1e-6)

            # train
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

                # Save loss values
                loss_val = loss.cpu().detach().numpy()
                loss_list.append(loss_val)

                # save the model with the minimum training loss
                if loss_val < min_loss:
                    torch.save(model, output_path + 'network.pkl')
                    min_loss = loss_val
                    
                # Save error
                if (it + 1) % (epochs/20/error_param) == 0:
                    u_L2RE = model.rel_error()
                    error_list.append(u_L2RE)
                    if record_time:
                        current_time = time.time()
                        time_messsage.append([current_time - start, loss_val, u_L2RE])

                # Output
                if (it + 1) % (epochs/20) == 0:
                    print("It = %d, loss = " % (it + 1), loss_val,", u_L2RE = ", u_L2RE, ", finish: %d%%" % ((it + 1) / epochs * 100))
                    
            end = time.time()
            print("Total train time: %.2fs" % (end - start))
            loss_list = np.array(loss_list).flatten()
            np.savetxt(output_path + "loss.txt", loss_list, fmt="%s",delimiter=' ')
            min_loss = np.min(loss_list)
            np.savetxt(output_path + "error.txt", error_list, fmt="%s",delimiter=' ')
            print("Min train loss: %.8f" % min_loss)
            if record_time:
                np.savetxt(output_path + "time_message.txt", time_messsage, fmt="%s",delimiter=' ')
        torch.cuda.empty_cache() # Release GPU memory
        
        # save loss curve
        loss_list = np.loadtxt(output_path + "loss.txt", dtype = float, delimiter=' ')
        plt.semilogy(loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if align:
            plt.ylim(lb_loss, ub_loss)
            plt.savefig(output_path + 'loss_aligned.pdf', format="pdf", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(output_path + 'loss.pdf', format="pdf", dpi=300, bbox_inches="tight")

        # Save loss curve
        fig, ax = plt.subplots()
        error_list = np.loadtxt(output_path + "error.txt", dtype = float, delimiter=' ')
        plt.semilogy(error_list)
        plt.xlabel('Epoch')
        plt.ylabel('Relative error')
        tool = [0, 4, 8, 12, 16, 20]
        plt.xticks([tool[i] * error_param for i in range(len(tool))], [str(0), str(int(epochs * 0.2)), str(int(epochs * 0.4)), str(int(epochs * 0.6)), str(int(epochs * 0.8)), str(int(epochs))])
        if align:
            plt.ylim(lb_error, ub_error) 
            plt.savefig(output_path + 'error_aligned.pdf', format="pdf", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(output_path + 'error.pdf', format="pdf", dpi=300, bbox_inches="tight")
            
        # record time related information
        if record_time:
            time_message = np.loadtxt(output_path + "time_message.txt", dtype = float, delimiter=' ')
            # time-loss curve
            fig, ax = plt.subplots()
            plt.semilogy(time_message[:,0], time_message[:,1])
            plt.xlabel('Time')
            plt.ylabel('Loss')
            if align:
                plt.ylim(lb_loss, ub_loss) 
                plt.savefig(output_path + 'time_loss_aligned.pdf', format="pdf", dpi=100, bbox_inches="tight")
            else:
                plt.savefig(output_path + 'time_loss.pdf', format="pdf", dpi=100, bbox_inches="tight")
                
            # time-error curve
            fig, ax = plt.subplots()
            plt.semilogy(time_message[:,0], time_message[:,2])
            plt.xlabel('Time')
            plt.ylabel('Relative error')
            if align:
                plt.ylim(lb_loss, ub_loss)
                plt.savefig(output_path + 'time_error_aligned.pdf', format="pdf", dpi=100, bbox_inches="tight")
            else:
                plt.savefig(output_path + 'time_error.pdf', format="pdf", dpi=100, bbox_inches="tight")

        # Calculate relative error
        dim = dim_test
        X = np.linspace(lb_geom, ub_geom, dim)
        Y = np.linspace(lb_geom, ub_geom, dim)
        X, Y = np.meshgrid(X, Y)
        X = X.flatten().reshape(dim*dim,1)
        Y = Y.flatten().reshape(dim*dim,1)
        X_pred = np.c_[X,Y]
        X_pred = torch.from_numpy(X_pred).float().to(device)
        u_truth = solution(X.flatten(),Y.flatten())
        model2 = torch.load(output_path + 'network.pkl', map_location=device) 
        u_pred = model2.predict(X_pred).flatten()
        u_L2RE = np.linalg.norm((u_truth - u_pred),ord=2) / (np.linalg.norm(u_truth, ord=2))
        print("u_L2RE = %.8f" % u_L2RE)
        np.savetxt(output_path + "u_truth.txt", u_truth, fmt="%s", delimiter=' ')
        np.savetxt(output_path + "u_pred.txt", u_pred, fmt="%s", delimiter=' ')

        # Plot predicted solution
        fig, ax = plt.subplots()
        if align:
                levels = np.arange(lb_u, ub_u + 1e-8, (ub_u - lb_u) / 100)
        else:
            levels = np.arange(min(u_pred) - abs(max(u_pred) - min(u_pred)) / 10, max(u_pred) + abs(max(u_pred) - min(u_pred)) / 10, (max(u_pred) - min(u_pred)) / 100) 
        cs = ax.contourf(X.reshape(dim,dim), Y.reshape(dim,dim), u_pred.reshape(dim,dim), levels,cmap='jet')
        cbar = fig.colorbar(cs)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('$u$(BsPINN)')
        if align:
            plt.savefig(output_path + "highhe_u_bspinn.png", format="png", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(output_path + "u_pred.png", format="png", dpi=300, bbox_inches="tight")
            
        # Plot exact solution
        fig, ax = plt.subplots()
        if align:
                levels = np.arange(lb_u, ub_u + 1e-8, (ub_u - lb_u) / 100)
        else:
            levels = np.arange(min(u_truth) - abs(max(u_truth) - min(u_truth)) / 10, max(u_truth) + abs(max(u_truth) - min(u_truth)) / 10, (max(u_truth) - min(u_truth)) / 100) 
        cs = ax.contourf(X.reshape(dim,dim), Y.reshape(dim,dim), u_truth.reshape(dim,dim), levels,cmap='jet')
        cbar = fig.colorbar(cs)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('$u$(truth)')
        if align:
            plt.savefig(output_path + "u_truth_aligned.png", format="png", dpi=150, bbox_inches="tight")
        else:
            plt.savefig(output_path + "u_truth.png", format="png", dpi=150, bbox_inches="tight")
        
        # Save error values
        csv_data.append([seed, u_L2RE])
    
    # Save as csv file
    output_path = path + ('./output_BsPINN/%s/' % name)
    file_write = open(output_path + 'record.csv','w')
    writer = csv.writer(file_write)
    writer.writerow(['seed','L2RE'])
    writer.writerows(csv_data)
    file_write.close()