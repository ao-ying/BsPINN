import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import time
import torch
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from torch.nn.parameter import Parameter
from vtk_to_files import var_to_polyvtk
from modulus.geometry.tessellation import Tessellation

# System settings
torch.set_default_dtype(torch.float)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
path = os.path.dirname(__file__) + "/"
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 20

# exact solution
def solution(x,y,z):
    return np.sin(kappa * x) * np.sin(kappa * y) * np.sin(kappa * z)

# partial derivative
def grad(f,x):
    ret = torch.autograd.grad(f, x, torch.ones_like(f).to(device), retain_graph=True, create_graph=True)[0]
    return ret

# Output model parameter information
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.params)
    trainable_num = sum(p.numel() for p in net.params if p.requires_grad)
    return trainable_num

# generate dataset
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

class PINN(nn.Module):
    def __init__(self, Layers, kappa, num_domain, boundary_weight, lb_X, ub_X):
        super(PINN, self).__init__()
        # initialize parameters
        self.mode = "train"
        self.kappa = kappa
        self.lb_X = lb_X
        self.ub_X = ub_X
        self.Layers = Layers
        self.num_domain = num_domain
        self.boundary_weight = boundary_weight
        self.params = []
        # Initialize binary structured neural network parameters
        self.weights, self.biases, self.w_last, self.b_last =  self.initialize_NN(self.Layers)
        
    # Initialize binary structured neural network parameters
    def initialize_NN(self, Layers):               
        num_Layers = len(Layers)       
        weights = [[] for i in range(num_Layers - 2)]
        biases = [[] for i in range(num_Layers - 2)]
        for l in range(0,num_Layers-2):
            for i in range(int(pow(2,l))):
                tempw = torch.zeros(Layers[l], Layers[l+1]).to(device)
                w = Parameter(tempw, requires_grad=True)
                nn.init.xavier_uniform_(w, gain=1)  
                tempb = torch.zeros(1, Layers[l+1]).to(device)
                b = Parameter(tempb, requires_grad=True)
                weights[l].append(w)
                biases[l].append(b) 
                self.params.append(w)
                self.params.append(b)
        tempw_last = torch.zeros(Layers[-2] * pow(2,num_Layers - 3), Layers[-1]).to(device)
        w_last = Parameter(tempw_last, requires_grad=True)
        tempb_last = torch.zeros(1, Layers[-1]).to(device)
        b_last = Parameter(tempb_last, requires_grad=True)
        self.params.append(w_last)
        self.params.append(b_last)  
        return weights, biases, w_last, b_last
    
    # binary structured neural network
    def neural_net(self, X):
        # Data preprocessing
        if torch.is_tensor(X) != True:         
            X = torch.from_numpy(X)
        ub = torch.from_numpy(self.ub_X).float().to(device)
        lb = torch.from_numpy(self.lb_X).float().to(device)  
        X = 2.0 * (X - lb) / (ub - lb) - 1.0
        H = X.float()
        
        if self.mode == "test": # If you want to use the CPU, you must include the following code
            self.weights = [[self.weights[i][j].to(device) for j in range(len(self.weights[i]))] for i in range(len(self.weights))]
            self.biases = [[self.biases[i][j].to(device) for j in range(len(self.biases[i]))] for i in range(len(self.biases))]
            self.w_last2 = Parameter(self.w_last.to(device))
            self.b_last2 = Parameter(self.b_last.to(device))

        # the first hidden layer
        num_Layers = len(self.Layers)
        l_out = torch.sin(torch.add(torch.matmul(H, self.weights[0][0]), self.biases[0][0]))
        temp = [[] for i in range(num_Layers - 2)]# tempSave the outputs of each hidden layer.
        temp[0].append(l_out)
        # the following hidden layer
        for l in range(1,num_Layers-2):
            for i in range(int(pow(2,l))):
                W = self.weights[l][i]
                b = self.biases[l][i]
                l_out = torch.sin(torch.add(torch.matmul(temp[l-1][int(i/2)], W), b))
                temp[l].append(l_out)
        # the last hidden layer
        out = temp[num_Layers - 3][0]
        for i in range(1, len(temp[num_Layers - 3])):
            out = torch.concat([out,temp[num_Layers - 3][i]],1)
        if self.mode == "train":
            Y = torch.add(torch.matmul(out, self.w_last), self.b_last)
        elif self.mode == "test":
            Y = torch.add(torch.matmul(out, self.w_last2), self.b_last2)
        
        return Y

    # PDE part
    def he_net(self, x, y, z):
        # governing equation
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
        dvalue = Y_train[self.num_domain:]
        boundary = u_b - dvalue

        return equation, boundary

    # loss function
    def loss(self,X_train):
        # governing equation and boundary condition
        x = X_train[:,0:1]
        y = X_train[:,1:2]
        z = X_train[:,2:3]
        equation, boundary = self.he_net(x, y, z)
        
        # total loss
        self.loss_e = torch.mean(torch.square(equation))
        self.loss_b = torch.mean(torch.square(boundary))
        loss_all = self.loss_e + self.boundary_weight * self.loss_b

        return loss_all
    
    # Predict the value of the displacement u at the corresponding point of X
    def predict(self, X):
        u_pred = self.neural_net(X)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred

    # set mode, train or test
    def set_mode(self, mode):
        self.mode = mode
    
# main
if __name__ == "__main__":
    seeds = [11]
    for seed in seeds:
        # seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # parameter settings
        ## Neural network related
        name = "helmholtz_3d_512-32"
        num_domain = 40000
        num_boundary = 4000
        num_domain_test = 10000
        num_boundary_test = 10000
        epochs = 5000 # Number of Adam optimizer iterations
        Layers = [3, 512, 256, 128, 64, 32, 1] # binary structured neural network structure
        learning_rate = 0.001 
        shuffle = False
        boundary_weight = 1
        patience = max(10, epochs/10)
        weight_decay = 0
        min_lr = 0
        X_train_interior_path = path + "../stl_model/interior_1e5.txt" 
        X_train_boundary_path = path + "../stl_model/boundary_1e7.txt"
        train = True
        align = False
        lb_loss = 1e-2
        ub_loss = 1e3
        
        ## equation related
        kappa = 8 * np.pi / 150 # The number of waves on a line segment with a length of 2pi. A complete cycle of waves is counted as one wave.
        
        # Auxiliary variables
        name = name + ("_%d" % epochs)
        print("***** name = %s *****" % name)
        print("seed = %d" % seed)
        output_path = path + ('../output/%s/' % name)
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('../output/%s/train_%d/' % (name, seed))
        if not os.path.exists(output_path): os.mkdir(output_path)

        # train
        if train:
            # generate train set
            print("Generating train set!")
            X_train_np, Y_train_np = load_data_file(X_train_interior_path, X_train_boundary_path, num_domain, num_boundary)
            lb_X = X_train_np.min(0)
            ub_X = X_train_np.max(0)
            X_train = torch.from_numpy(X_train_np).float().to(device)
            Y_train = torch.from_numpy(Y_train_np).float().to(device)
            
            # generate test set
            print("Generating test set!")
            X_test_np, Y_test_np = load_data_file(X_train_interior_path, X_train_boundary_path, num_domain_test, num_boundary_test)
            X_test = torch.from_numpy(X_test_np).float().to(device)
            Y_test = torch.from_numpy(Y_test_np).float().to(device)

            # Declare neural network instance
            model = PINN(Layers, kappa, num_domain, boundary_weight, lb_X, ub_X)
            model = nn.DataParallel(model)
            model = model.module
            model.to(device)
            print(model) # Print network summary
            
            # Adam optimizer
            optimizer = optim.Adam(model.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, verbose=True, min_lr=min_lr)

            # train
            start = time.time()
            loss_list = []
            min_train_loss = 9999999
            print("Start training")
            for it in range(epochs):
                # Optimizer training
                loss = model.loss(X_train) 
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                # decay learning rate
                scheduler.step(loss)

                # save loss value
                loss_val = loss.cpu().detach().numpy()
                loss_list.append(loss_val)

                # save the model with the minimum training loss value
                if loss_val < min_train_loss:
                    torch.save(model, output_path + 'network.pkl')
                    min_train_loss = loss_val

                # Print intermediate results
                if (it + 1) % (epochs/20) == 0:
                    # calculate relative error
                    model2 = torch.load(output_path + 'network.pkl') 
                    u_pred = model2.predict(X_test).flatten()
                    u_truth = Y_test_np.flatten()
                    u_error = np.linalg.norm((u_truth - u_pred),ord=2) / (np.linalg.norm(u_truth, ord=2))
                    # print
                    print("It = %d, loss = %.8f, u_error = %.8f, finish: %d%%" % ((it + 1), loss_val, u_error, ((it + 1) / epochs * 100)))
                    print("loss_e = %.8f, loss_b = %.8f" % (model.loss_e, model.loss_b))
            # following processing
            end = time.time()
            train_time = end - start
            loss_list = np.array(loss_list).flatten()
            min_loss = np.min(loss_list)
            np.savetxt(output_path + "loss.txt", loss_list, fmt="%s",delimiter=' ')
            params = get_parameter_number(model)
            print("time = %.2fs" % train_time)
            print("params = %d" % params)
            print("min_loss = %.8f" % min_loss)
        else:
            print("Generating test set!")
            X_test_np, Y_test_np = load_data_file(X_train_interior_path, X_train_boundary_path, num_domain_test, num_boundary_test)
            X_test = torch.from_numpy(X_test_np).float().to(device)
            Y_test = torch.from_numpy(Y_test_np).float().to(device)

        # save loss curve
        loss_list = np.loadtxt(output_path + "loss.txt", dtype = float, delimiter=' ')
        fig, ax = plt.subplots()
        plt.semilogy(loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if align:
            plt.ylim(lb_loss, ub_loss) 
            plt.savefig(output_path + 'loss_aligned.pdf', format="pdf", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(output_path + 'loss.pdf', format="pdf", dpi=300, bbox_inches="tight")

        # calculate u_pred and u_truth
        u_truth = solution(X_test_np[:,0], X_test_np[:,1], X_test_np[:,2])
        model2 = torch.load(output_path + 'network.pkl', map_location=device) 
        model2.set_mode('test') # Be sure to change to test mode when predicting on CPU
        u_pred = model2.predict(X_test).flatten()
        u_mae = np.mean(np.abs(u_truth - u_pred))
        print("u_mae = %.8f" % u_mae)
        u_error = np.linalg.norm((u_truth - u_pred),ord=2) / (np.linalg.norm(u_truth, ord=2))
        print("u_error = %.8f" % u_error)

        # save u_pred to a vtp file
        u_vtp = {'x': X_test_np[:, 0:1], 'y': X_test_np[:, 1:2], 'z': X_test_np[:, 2:3], 'u_pred': u_pred}
        var_to_polyvtk(u_vtp, output_path + "u_pred_bfc_boundary4e6") 