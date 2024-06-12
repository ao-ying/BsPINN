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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_cpu = torch.device('cpu')
torch.backends.cudnn.benchmark = True
path = os.path.dirname(__file__) + "/"
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 20

# exact solution
def solution(x,y,z):
    return np.sin(kappa * x) * np.sin(kappa * y) * np.sin(kappa * z)

# Compute partial derivatives
def grad(f,x):
    ret = torch.autograd.grad(f, x, torch.ones_like(f).to(device), retain_graph=True, create_graph=True)[0]
    return ret

# Output model parameter information
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.params)
    trainable_num = sum(p.numel() for p in net.params if p.requires_grad)
    return trainable_num

# Import datasets from file
def load_data_file(X_train_interior_path, X_train_boundary_path, num_domain, num_boundary):
    interior = np.loadtxt(X_train_interior_path, dtype = float, delimiter=' ')
    np.random.shuffle(interior)
    interior = interior[0:num_domain]
    boundary = np.loadtxt(X_train_boundary_path, dtype = float, delimiter=' ')
    np.random.shuffle(boundary)
    boundary = boundary[0:num_boundary]
    X_train = np.concatenate([interior, boundary], 0)
    Y_train = solution(X_train[:,0], X_train[:,1], X_train[:,2])
    Y_train = Y_train.reshape(Y_train.shape[0], 1)

    return X_train, Y_train


class BsPINN(nn.Module):
    def __init__(self, Layers, kappa, num_domain, boundary_weight, lb_X, ub_X):
        super(BsPINN, self).__init__()
        # Initialize parameters
        self.kappa = kappa
        self.lb_X = lb_X
        self.ub_X = ub_X
        self.Layers = Layers
        self.num_domain = num_domain
        self.boundary_weight = boundary_weight
        self.params = []
        self.act = torch.sin
        self.num_Layers = len(self.Layers)
        # Related to binary network
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
    
    # Create masks
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
    
    # Initialize binary network parameters
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
            # Weights w
            tempw = torch.zeros(self.width[l], self.width[l+1]).to(device)
            # Block diagonal matrix initialization
            for i in range(int(pow(2,l))): # Iterate through each submatrix and initialize
                tempw2 = torch.zeros(Layers[l], Layers[l+1])
                w2 = Parameter(tempw2, requires_grad=True)
                nn.init.xavier_uniform_(w2, gain=1)  
                row_index = int(i / 2)
                tempw[row_index * Layers[l] : (row_index + 1) * Layers[l], i * Layers[l+1] : (i + 1) * Layers[l+1]] = w2.data
            w = Parameter(tempw, requires_grad=True)
            # Biases b
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
    
    # Binary network part
    def neural_net(self, X):
        # Data preprocessing
        X = 2.0 * (X - self.lb_X) / (self.ub_X - self.lb_X) - 1.0
        X = X.float()

        # First hidden layer
        for l in range(0, self.num_Layers - 2):
            if l >=2 and l <= self.num_Layers - 3:
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
    def he_net(self, x, y, z):
        # Equation
        x_e = x[0:self.num_domain].clone()
        x_e = x_e.requires_grad_()
        y_e = y[0:self.num_domain].clone()
        y_e = y_e.requires_grad_()
        z_e = z[0:self.num_domain].clone()
        z_e = z_e.requires_grad_()
        u_e = self.neural_net(torch.cat([x_e,y_e,z_e], dim = 1))
        f = 2 * self.kappa ** 2 * torch.sin(self.kappa * x_e) * torch.sin(self.kappa * y_e) * torch.sin(self.kappa * z_e)
        u_x = grad(u_e, x_e)
        u_xx = grad(u_x, x_e)
        u_y = grad(u_e, y_e)
        u_yy = grad(u_y, y_e)
        u_z = grad(u_e, z_e)
        u_zz = grad(u_z, z_e)
        equation = - u_xx - u_yy - u_zz - self.kappa ** 2 * u_e - f

        # Dirichlet boundary conditions
        x_b = x[self.num_domain:].clone()
        y_b = y[self.num_domain:].clone()
        z_b = z[self.num_domain:].clone()
        u_b = self.neural_net(torch.cat([x_b, y_b, z_b], dim = 1))
        # dvalue = torch.sin(self.kappa * x_b) * torch.sin(self.kappa * y_b) * torch.sin(self.kappa * z_b)
        dvalue = Y_train[self.num_domain:]
        boundary = u_b - dvalue

        return equation, boundary

    # Loss function
    def loss(self,X_train):
        # Calculate equation and boundary terms
        x = X_train[:,0:1]
        y = X_train[:,1:2]
        z = X_train[:,2:3]
        equation, boundary = self.he_net(x, y, z)
        
        # Calculate total error
        self.loss_e = torch.mean(torch.square(equation))
        self.loss_b = torch.mean(torch.square(boundary))
        loss_all = self.loss_e + self.boundary_weight * self.loss_b

        return loss_all
    
    # Predict u value at point X
    def predict(self, X):
        u_pred = self.neural_net(X)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred
    
# main function
if __name__ == "__main__":
    seeds = [13]
    csv_data = []
    for seed in seeds:
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        csv_data = []
        # Parameter settings
        ## Neural network related
        train = True
        name = "BsPINN_512-32_moredata" 
        epochs = 20000 # Number of iterations for adam optimizer
        num_domain = 60000
        num_boundary = 5000 
        num_domain_test = 10000 
        num_boundary_test = 10000 
        Layers = [3, 512, 256, 128, 64, 32, 1] # Binary network structure
        learning_rate = 0.001 # Learning rate decay strategy to be set
        shuffle = False # Shuffle data at the start of each epoch
        boundary_weight = 1
        patience = max(10, epochs/10)
        weight_decay = 0
        min_lr = 0
        X_train_interior_path = path + "../stl_model/interior_1e5.txt"
        X_train_boundary_path = path + "../stl_model/boundary_1e5.txt"
        
        ## Plot related
        align = False
        lb_loss = 1e-5
        ub_loss = 1e2
        lb_error = 0.1
        ub_error = 1.1
        error_param = 1 # 5 # Plot (20 * error_param) error points, epochs should be divisible by (20 * error_param).
        
        ## Equation related
        kappa = 8 * np.pi / 150 # Number of waves on a segment of length 2pi, one complete cycle counts as one wave. Corresponds to the parameter omega in the paper, the larger the omega, the more intense the fluctuation.
        
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

        # Training
        if train:
            # Generate training set
            print("Generating train set!")
            X_train_np, Y_train_np = load_data_file(X_train_interior_path, X_train_boundary_path, num_domain, num_boundary)
            lb_X = X_train_np.min(0)
            ub_X = X_train_np.max(0)
            lb_X = torch.from_numpy(lb_X).float().to(device)
            ub_X = torch.from_numpy(ub_X).float().to(device)
            X_train = torch.from_numpy(X_train_np).float().to(device)
            Y_train = torch.from_numpy(Y_train_np).float().to(device)
            
            # Generate test set
            print("Generating test set!")
            X_test_np, Y_test_np = load_data_file(X_train_interior_path, X_train_boundary_path, num_domain_test, num_boundary_test)
            X_test = torch.from_numpy(X_test_np).float().to(device)
            Y_test = torch.from_numpy(Y_test_np).float().to(device)

            # Declare neural network instance
            model = BsPINN(Layers, kappa, num_domain, boundary_weight, lb_X, ub_X)
            model = nn.DataParallel(model)
            model = model.module
            model.to(device)
            print(model) # Print network summary
            params = model.num_param
            print("params = %d" % params)
            
            # Adam optimizer
            optimizer = optim.Adam(model.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, verbose=True, min_lr=min_lr)

            # Training
            start = time.time()
            loss_list = []
            error_list = [] 
            min_train_loss = 9999999
            print("Start training")
            for it in range(epochs):
                # Optimizer training
                loss = model.loss(X_train) 
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                # Learning rate decay
                scheduler.step(loss)

                # Save loss values
                loss_val = loss.cpu().detach().numpy()
                loss_list.append(loss_val)

                # Save the model when the training error is minimized
                if loss_val < min_train_loss:
                    torch.save(model, output_path + 'network.pkl')
                    min_train_loss = loss_val
                    
                # Save error curve
                if (it + 1) % (epochs/20/error_param) == 0:
                    # Note that model should be used to calculate the error here
                    u_pred = model.predict(X_test).flatten()
                    u_truth = Y_test_np.flatten()
                    u_L2RE = np.linalg.norm((u_truth - u_pred),ord=2) / (np.linalg.norm(u_truth, ord=2))
                    error_list.append(u_L2RE)

                # Output
                if (it + 1) % (epochs/20) == 0:
                    # Output
                    print("It = %d, loss = %.8f, u_L2RE = %.8f, finish: %d%%" % ((it + 1), loss_val, u_L2RE, ((it + 1) / epochs * 100)))
                    # print("loss_e = %.8f, loss_b = %.8f" % (model.loss_e, model.loss_b))
            # Post-processing
            end = time.time()
            train_time = end - start
            loss_list = np.array(loss_list).flatten()
            min_loss = np.min(loss_list)
            np.savetxt(output_path + "loss.txt", loss_list, fmt="%s",delimiter=' ')
            np.savetxt(output_path + "error.txt", error_list, fmt="%s",delimiter=' ')
            print("time = %.2fs" % train_time)
            print("min_loss = %.8f" % min_loss)
        else:
            # This step is to ensure that the test set is consistent with the training.
            X_train_np, Y_train_np = load_data_file(X_train_interior_path, X_train_boundary_path, num_domain, num_boundary)
            print("Generating test set!")
            X_test_np, Y_test_np = load_data_file(X_train_interior_path, X_train_boundary_path, num_domain_test, num_boundary_test)
            X_test = torch.from_numpy(X_test_np).float().to(device)
            Y_test = torch.from_numpy(Y_test_np).float().to(device)

        # Save loss curve
        loss_list = np.loadtxt(output_path + "loss.txt", dtype = float, delimiter=' ')
        fig, ax = plt.subplots()
        plt.semilogy(loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if align:
            plt.ylim(lb_loss, ub_loss) # Unified scale with BiPINN
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
            plt.ylim(lb_error, ub_error) # Unified scale with BiPINN
            plt.savefig(output_path + 'error_aligned.pdf', format="pdf", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(output_path + 'error.pdf', format="pdf", dpi=300, bbox_inches="tight")

        # Calculate u_pred and u_truth
        u_truth = solution(X_test_np[:,0], X_test_np[:,1], X_test_np[:,2])
        model2 = torch.load(output_path + 'network.pkl', map_location=device) 
        u_pred = model2.predict(X_test).flatten()
        u_mae = np.mean(np.abs(u_truth - u_pred))
        print("u_mae = %.8f" % u_mae)
        u_L2RE = np.linalg.norm((u_truth - u_pred),ord=2) / (np.linalg.norm(u_truth, ord=2))
        print("u_L2RE = %.8f" % u_L2RE)
        
        # Save error values
        csv_data.append([seed, u_L2RE])
    
    # Save as csv file
    output_path = path + ('./output_BsPINN/%s/' % name)
    file_write = open(output_path + 'record.csv','w')
    writer = csv.writer(file_write)
    writer.writerow(['seed','L2RE'])
    writer.writerows(csv_data)
    file_write.close()