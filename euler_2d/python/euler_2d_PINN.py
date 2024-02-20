import os
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

# partial derivative
def grad(f,x):
    ret = torch.autograd.grad(f, x, torch.ones_like(f).to(device), retain_graph=True, create_graph=True)[0]
    return ret

# output model parameter information
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.params)
    trainable_num = sum(p.numel() for p in net.params if p.requires_grad)
    return trainable_num

# Lists are accumulated element by element to form a new list
def accsum(l):
    ret = [l[0]]
    temp = l[0]
    for i in range(1,len(l)):
        temp += l[i]
        ret.append(temp)
    return ret

# the exact solution of density
def sol_r(x,y,t):
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

# generate dataset
def load_data(N0, Nb, Nf, lb_x, ub_x, lb_y, ub_y, lb_t, ub_t):
    # governing equation
    X_train = np.random.uniform([lb_x, lb_y, lb_t], [ub_x, ub_y, ub_t], (Nf, 3))
    # initial condition
    points = np.random.uniform([lb_x, lb_y, lb_t], [ub_x, ub_y, lb_t], (N0, 3))
    X_train = np.r_[X_train, points]
    # boundary condition
    points = np.random.uniform([lb_x, lb_y, lb_t], [lb_x, ub_y, ub_t], (Nb, 3)) # left
    X_train = np.r_[X_train, points]
    points = np.random.uniform([ub_x, lb_y, lb_t], [ub_x, ub_y, ub_t], (Nb, 3)) # right
    X_train = np.r_[X_train, points]
    points = np.random.uniform([lb_x, lb_y, lb_t], [ub_x, lb_y, ub_t], (Nb, 3)) # down
    X_train = np.r_[X_train, points]
    points = np.random.uniform([lb_x, ub_y, lb_t], [ub_x, ub_y, ub_t], (Nb, 3)) # up
    X_train = np.r_[X_train, points]
    
    # calculate true value of density
    r_train = sol_r(X_train[:,0], X_train[:,1], X_train[:,2])
    r_train = np.reshape(r_train, (r_train.shape[0], 1))
    
    return X_train, r_train

class PINN(nn.Module):
    def __init__(self, Layers, N0, Nb, Nf, initial_weight, boundary_weight, lb_X, ub_X, gamma, r_train):
        super(PINN, self).__init__()
        # initialize parameters
        self.iter = 0
        self.lb_X = lb_X
        self.ub_X = ub_X
        self.Layers = Layers
        self.r_train = r_train
        self.gamma = gamma
        self.Nf = Nf
        self.N0 = N0
        self.Nb = Nb
        self.lens = accsum([Nf, N0, Nb, Nb, Nb, Nb])
        self.initial_weight = initial_weight
        self.boundary_weight = boundary_weight
        self.loss_function = nn.MSELoss(reduction ='mean')
        self.params = []
        # Initialize fully connected neural network parameters
        self.weights, self.biases =  self.initialize_NN(self.Layers)
        
    # Initialize fully connected neural network parameters
    def initialize_NN(self, Layers):               
        num_Layers = len(Layers)       
        weights = []
        biases = []
        for l in range(0,num_Layers-1):
            tempw = torch.zeros(Layers[l], Layers[l+1]).to(device)
            w = Parameter(tempw, requires_grad=True)
            nn.init.xavier_uniform_(w, gain=1) # normal
            tempb = torch.zeros(1, Layers[l+1]).to(device)
            b = Parameter(tempb, requires_grad=True)
            weights.append(w)
            biases.append(b) 
            self.params.append(w)
            self.params.append(b)
        return weights, biases
    
    # fully connected neural network
    def neural_net(self, X):
        # Data preprocessing
        X = 2.0 * (X - self.lb_X) / (self.ub_X - self.lb_X) - 1.0
        H = X.float()
        
        # forward propagation
        num_Layers = len(self.Layers)
        for l in range(0,num_Layers-2):
            W = self.weights[l]
            b = self.biases[l]
            H = torch.tanh(torch.add(torch.matmul(H, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        Y = torch.add(torch.matmul(H, W), b)
        return Y

    # PDE part
    def he_net(self, X):
        # governing equation
        x_e = X[0 : self.lens[0], 0:1].clone()
        x_e = x_e.requires_grad_()
        y_e = X[0 : self.lens[0], 1:2].clone()
        y_e = y_e.requires_grad_()
        t_e = X[0 : self.lens[0], 2:3].clone()
        t_e = t_e.requires_grad_()
        out_e = self.neural_net(torch.cat([x_e, y_e, t_e], dim = 1))
        r_e = out_e[:,0:1]
        u_e = out_e[:,1:2]
        v_e = out_e[:,2:3]
        p_e = out_e[:,3:4]
        E_e = out_e[:,4:5]
        equation1 = grad(r_e, t_e) + grad(r_e * u_e, x_e) + grad(r_e * v_e, y_e)
        equation2 = grad(r_e * u_e, t_e) + grad(r_e * u_e**2 + p_e, x_e) + grad(r_e * u_e * v_e, y_e)
        equation3 = grad(r_e * v_e, t_e) + grad(r_e * u_e * v_e, x_e) + grad(r_e * v_e**2 + p_e, y_e) 
        equation4 = grad(r_e * E_e, t_e) + grad(u_e * (r_e * E_e + p_e), x_e) + grad(v_e * (r_e * E_e + p_e), y_e)
        equation5 = p_e - (self.gamma - 1) * (r_e * E_e - 0.5 * r_e * (u_e**2 + v_e**2))
        equations = [equation1, equation2, equation3, equation4, equation5]
        
        # initial condition
        initials = []
        out_i1 = self.neural_net(X[self.lens[0]: self.lens[1], :])
        rval = self.r_train[self.lens[0]: self.lens[1]]
        r_i = out_i1[:,0:1]
        u_i = out_i1[:,1:2]
        v_i = out_i1[:,2:3]
        p_i = out_i1[:,3:4]
        initials.extend([r_i - rval, u_i - 0.1, v_i - 0, p_i - 1.0])
        
        # boundary condition
        boundarys = []
        ## left
        out_b1 = self.neural_net(X[self.lens[1]: self.lens[2], :])
        rval = self.r_train[self.lens[1]: self.lens[2]]
        r_b = out_b1[:,0:1]
        u_b = out_b1[:,1:2]
        v_b = out_b1[:,2:3]
        p_b = out_b1[:,3:4]
        boundarys.extend([r_b - rval, u_b - 0.1, v_b - 0, p_b - 1.0])
        ## right
        out_b2 = self.neural_net(X[self.lens[2]: self.lens[3], :])
        rval = self.r_train[self.lens[2]: self.lens[3]]
        r_b = out_b2[:,0:1]
        u_b = out_b2[:,1:2]
        v_b = out_b2[:,2:3]
        p_b = out_b2[:,3:4]
        boundarys.extend([r_b - rval, u_b - 0.1, v_b - 0, p_b - 1.0])
        ## up
        out_b3 = self.neural_net(X[self.lens[3]: self.lens[4], :])
        rval = self.r_train[self.lens[3]: self.lens[4]]
        r_b = out_b3[:,0:1]
        u_b = out_b3[:,1:2]
        v_b = out_b3[:,2:3]
        p_b = out_b3[:,3:4]
        boundarys.extend([r_b - rval, u_b - 0.1, v_b - 0, p_b - 1.0])
        ## down
        out_b4 = self.neural_net(X[self.lens[4]: self.lens[5], :])
        rval = self.r_train[self.lens[4]: self.lens[5]]
        r_b = out_b4[:,0:1]
        u_b = out_b4[:,1:2]
        v_b = out_b4[:,2:3]
        p_b = out_b4[:,3:4]
        boundarys.extend([r_b - rval, u_b - 0.1, v_b - 0, p_b - 1.0])
        
        return equations, initials, boundarys

    # loss function
    def loss(self,X_train):
        # governing equation, initial conditon, boundary condition
        equations, initials, boundarys = self.he_net(X_train)
        
        # total loss
        loss_e = 0
        for i in range(len(equations)): loss_e += torch.mean(torch.pow(equations[i], 2))
        loss_i = 0
        for i in range(len(initials)): loss_i += torch.mean(torch.pow(initials[i], 2))
        loss_b = 0
        for i in range(len(boundarys)): loss_b += torch.mean(torch.pow(boundarys[i], 2))
        loss_all = loss_e + self.initial_weight * loss_i + self.boundary_weight * loss_i

        return loss_all

    # Predict the value of the density u at the corresponding point of X
    def predict(self, X):
        u_pred = self.neural_net(X)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred 

# main
if __name__ == "__main__":
    seeds = [11]
    for seed in seeds:
        print("***** seed = %d *****" % seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # parameter settings
        ## equation related
        gamma = 1.4
        lb_x = 0
        ub_x = 1
        lb_y = 0
        ub_y = 1
        lb_t = 0
        ub_t = 2
        
        ## Neural network related
        name = "euler_2d_5*256"
        Nf = 10000 # number of training points corresponding to governing equations
        N0 = 400 # number of training points corresponding to initial condition
        Nb = 100 # number of training points corresponding to boundary condition on each edge
        epochs = 5000 # Number of Adam optimizer iterations
        Layers = [3,256,256,256,256,256,5] # fully connected neural network structure
        learning_rate = 0.001
        patience = max(10, epochs / 10) # patience for decaying learning rate
        initial_weight = 1 # initial weight
        boundary_weight = 1 # boundary weight
        weight_decay = 0 
        
        ## draw related
        train = True
        align = False
        lb_loss = 1e-5
        ub_loss = 1e4
        lb_r = 0.85
        ub_r = 1.6
        
        # Auxiliary variables
        name = name + ("_%d" % epochs)
        print("\n\n***** name = %s *****" % name)
        print("seed = %d" % seed)
        output_path = path + ('../output')
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('../output/%s' % name)
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('../output/%s/train_%d/' % (name, seed))
        if not os.path.exists(output_path): os.mkdir(output_path)
        
        if train:
            # generate train set
            print("Loading data")
            X_train, r_train = load_data(N0, Nb, Nf, lb_x, ub_x, lb_y, ub_y, lb_t, ub_t)
            lb_X = X_train.min(0) 
            ub_X = X_train.max(0) 
            lb_X = torch.from_numpy(lb_X).float().to(device)
            ub_X = torch.from_numpy(ub_X).float().to(device)
            X_train = torch.from_numpy(X_train).float().to(device)
            r_train = torch.from_numpy(r_train).float().to(device)
            
            # Declare neural network instance
            model = PINN(Layers, N0, Nb, Nf, initial_weight, boundary_weight, lb_X, ub_X, gamma, r_train)
            model = nn.DataParallel(model)
            model = model.module
            model.to(device)
            print(model) # Print network summary
            
            # Adam optimizer
            optimizer = optim.Adam(model.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, verbose=True, min_lr=1e-6) 
            
            # data preparation
            dim = 500
            X = np.linspace(lb_x, ub_x, dim)
            Y = np.linspace(lb_y, ub_y, dim)
            X,Y = np.meshgrid(X,Y)
            X = X.flatten().reshape(dim * dim, 1)
            Y = Y.flatten().reshape(dim * dim, 1)
            T = np.full((dim * dim, 1), 1.0)
            r_truth = sol_r(X.flatten(),Y.flatten(), T.flatten())
            X_pred= np.c_[X, Y, T] # N * 3
            X_pred = torch.from_numpy(X_pred).float().to(device)

            # train
            start = time.time()
            ## adam
            loss_list = []
            min_loss = 9999999
            print("Start Adam training!\n")
            for it in range(epochs):
                # Optimizer training
                Loss = model.loss(X_train)
                optimizer.zero_grad(set_to_none=True)
                Loss.backward() 
                optimizer.step()

                # decay learning rate
                scheduler.step(Loss)

                # save loss value and model with the minimum train loss
                loss_val = Loss.cpu().detach().numpy()
                loss_list.append(loss_val)
                if loss_val < min_loss: 
                    torch.save(model, output_path + 'network.pkl') 
                    min_loss = loss_val

                # print intermediate results
                if (it + 1) % (epochs / 20) == 0:
                    # calculate relative error
                    model2 = torch.load(output_path + 'network.pkl', map_location=device) 
                    pred = model2.predict(X_pred)
                    r_pred = pred[:,0]
                    r_error = np.linalg.norm((r_truth - r_pred),ord=2) / (np.linalg.norm(r_truth, ord=2))
                    # print
                    print("It = %d, loss = %.8f, r_error = %.8f, finish: %d%%" % ((it + 1), loss_val, r_error, ((it + 1) / epochs * 100)))
                
            # following processing
            end = time.time()
            train_time = end - start
            loss_list = np.array(loss_list).flatten()
            min_loss = np.min(loss_list)
            params = get_parameter_number(model)
            np.savetxt(output_path + "loss.txt", loss_list, fmt="%s",delimiter=' ')
            print("time = %.2fs" % train_time)
            print("params = %d" % params)
            print("min_loss = %.8f" % min_loss)

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
            
        # error assessment
        dim = 100
        points = np.loadtxt(path + "../data/points_%d.txt" % dim, dtype = float, delimiter=' ')
        X_pred = torch.from_numpy(points).float().to(device)
        model2 = torch.load(output_path + 'network.pkl', map_location=device) 
        pred = model2.predict(X_pred)
        r_pred = pred[:,0]
        r_truth = sol_r(points[:,0], points[:,1], points[:,2])
        r_error = np.linalg.norm((r_truth - r_pred),ord=2) / (np.linalg.norm(r_truth, ord=2))
        print("r_error = %.8f" % r_error)

        # Calculate the relative error of density on the xy-plane at t=1s
        dim = 500
        X = np.linspace(lb_x, ub_x, dim)
        Y = np.linspace(lb_y, ub_y, dim)
        X,Y = np.meshgrid(X,Y)
        X = X.flatten().reshape(dim * dim, 1)
        Y = Y.flatten().reshape(dim * dim, 1)
        T = np.full((dim * dim, 1), 1.0)
        X_pred= np.c_[X, Y, T] # N * 3
        X_pred = torch.from_numpy(X_pred).float().to(device)
        model2 = torch.load(output_path + 'network.pkl', map_location=device) 
        pred = model2.predict(X_pred)
        r_pred = pred[:,0]
        r_truth = sol_r(X.flatten(),Y.flatten(), T.flatten())
        r_error = np.linalg.norm((r_truth - r_pred),ord=2) / (np.linalg.norm(r_truth, ord=2))
        print("r_error = %.8f when drawing." % r_error)

        # draw the image of predicted solution
        fig, ax = plt.subplots()
        if align:
            levels = np.arange(lb_r, ub_r, (ub_r - lb_r) / 100)
        else:
            levels = np.arange(min(r_pred) - abs(max(r_pred) - min(r_pred)) / 10, max(r_pred) + abs(max(r_pred) - min(r_pred)) / 10, (max(r_pred) - min(r_pred)) / 100) 
        cs = ax.contourf(X.reshape(dim, dim), Y.reshape(dim, dim), r_pred.reshape(dim, dim), levels,cmap=plt.get_cmap('Spectral'))
        cbar = fig.colorbar(cs)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('$r$(PINN) when t=1s')
        if align:
            plt.savefig(output_path + "r_pred_aligned.png", format="png", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(output_path + "r_pred.png", format="png", dpi=300, bbox_inches="tight")
        
        # draw the image of exact solution
        fig, ax = plt.subplots()
        if align:
            levels = np.arange(lb_r, ub_r, (ub_r - lb_r) / 100)
        else:
            levels = np.arange(min(r_truth) - abs(max(r_truth) - min(r_truth)) / 10, max(r_truth) + abs(max(r_truth) - min(r_truth)) / 10, (max(r_truth) - min(r_truth)) / 100) 
        cs = ax.contourf(X.reshape(dim, dim), Y.reshape(dim, dim), r_truth.reshape(dim, dim), levels,cmap=plt.get_cmap('Spectral'))
        cbar = fig.colorbar(cs)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('$r$(truth) when t=1s')
        if align:
            plt.savefig(output_path + "r_truth_aligned.png", format="png", dpi=300, bbox_inches="tight")
        else:
            plt.savefig(output_path + "r_truth.png", format="png", dpi=300, bbox_inches="tight")