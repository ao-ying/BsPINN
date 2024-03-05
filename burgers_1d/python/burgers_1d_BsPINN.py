import os
import numpy as np
from matplotlib import pyplot as plt
import time
import torch
import torch.nn as nn         
import torch.optim as optim             
from torch.nn.parameter import Parameter
import scipy.io
from pyDOE import lhs

# System settings
torch.set_default_dtype(torch.float)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Device configuration
device_cpu = torch.device('cpu')
path = os.path.dirname(__file__) + "/"
torch.backends.cudnn.benchmark = True
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 30

# partial derivative
def grad(f,x):
    ret = torch.autograd.grad(f, x, torch.ones_like(f).to(device), retain_graph=True, create_graph=True)[0]
    return ret

# output model parameter information
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.params)
    trainable_num = sum(p.numel() for p in net.params if p.requires_grad)
    return trainable_num

# generate dataset
def load_data(N0, Nb, Nf, lb_x, ub_x, lb_t, ub_t):
    data = scipy.io.loadmat(path + '../data/burgers_shock.mat')
    t = np.linspace(lb_t, ub_t, 3 * Nb); t = np.reshape(t, (t.shape[0], 1))
    x = np.linspace(lb_x, ub_x, 3 * N0); x = np.reshape(x, (x.shape[0], 1))
    Exact = data['usol'] # exact solution of u, 256 * 100.
    u_truth = np.real(Exact).T.flatten()
    # governing equation
    X_train = [lb_x, lb_t] + [ub_x - lb_x, ub_t - lb_t] * lhs(2, Nf) # Nf * 2
    # initial condition
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:] # N0 * 1
    X0_train = np.c_[x0, np.full((N0, 1), lb_t)] # N0 * 2
    X_train = np.r_[X_train, X0_train] # (Nf + N0) * 2
    # boundary condition
    idx_t = np.random.choice(t.shape[0], Nb, replace=False) # Nb * 1
    tb = t[idx_t,:] # Nb * 1
    Xb_train = np.c_[np.full((Nb, 1), lb_x), tb]
    Xb_train = np.r_[Xb_train, np.c_[np.full((Nb, 1), ub_x), tb]]
    X_train = np.r_[X_train, Xb_train]
    return X_train, u_truth

class BsPINN(nn.Module):
    def __init__(self, Layers, N0, Nb, Nf, initial_weight, boundary_weight, lb_X, ub_X):
        super(BsPINN, self).__init__()
        # initialize parameters
        self.iter = 0
        self.lb_X = lb_X
        self.ub_X = ub_X
        self.Layers = Layers
        self.Nf = Nf
        self.N0 = N0
        self.Nb = Nb
        self.lens = [Nf, Nf + N0, Nf + N0 + Nb, Nf + N0 + 2 * Nb]
        self.initial_weight = initial_weight
        self.boundary_weight = boundary_weight
        self.params = []
        # Initialize the parameters of binary structured neural network 
        self.weights, self.biases, self.w_last, self.b_last =  self.initialize_NN(self.Layers)
        
    # Initialize the parameters of binary structured neural network 
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
        X = 2.0 * (X - self.lb_X) / (self.ub_X - self.lb_X) - 1.0
        H = X.float()
        
        # the first hidden layer
        num_Layers = len(self.Layers)
        l_out = torch.tanh(torch.add(torch.matmul(H, self.weights[0][0]), self.biases[0][0]))
        temp = [[] for i in range(num_Layers - 2)]# save the outputs of each hidden layer.
        temp[0].append(l_out)
        # the following hidden layer
        for l in range(1,num_Layers-2):
            for i in range(int(pow(2,l))):
                W = self.weights[l][i]
                b = self.biases[l][i]
                l_out = torch.tanh(torch.add(torch.matmul(temp[l-1][int(i/2)], W), b))
                temp[l].append(l_out)
        # the last hidden layer
        out = temp[num_Layers - 3][0]
        for i in range(1, len(temp[num_Layers - 3])):
            out = torch.concat([out,temp[num_Layers - 3][i]],1)
        Y = torch.add(torch.matmul(out, self.w_last),self. b_last)
        return Y
    
    def neural_net_channel(self, X, channel):
        # Data preprocessing
        X = 2.0 * (X - self.lb_X) / (self.ub_X - self.lb_X) - 1.0
        H = X.float()

        # the first hidden layer
        num_Layers = len(self.Layers)
        l_out = torch.sin(torch.add(torch.matmul(H, self.weights[0][0]), self.biases[0][0]))
        temp = [[] for i in range(num_Layers - 2)]# save the outputs of each hidden layer.
        temp[0].append(l_out)
        # the following hidden layer
        for l in range(1,num_Layers-2):
            for i in range(int(pow(2,l))):
                W = self.weights[l][i]
                b = self.biases[l][i]
                l_out = torch.sin(torch.add(torch.matmul(temp[l-1][int(i/2)], W), b))
                temp[l].append(l_out)
        # the last hidden layer
        out = torch.zeros(X.shape[0], self.Layers[1]).to(device)
        out[:, self.Layers[-2] * channel : self.Layers[-2] * (channel + 1)] = temp[num_Layers - 3][channel]
        Y = torch.add(torch.matmul(out, self.w_last),self. b_last)
        return Y

    # PDE part
    def he_net(self, X):
        # governing equation
        x_e = X[0 : self.lens[0], 0:1].clone()
        x_e = x_e.requires_grad_()
        t_e = X[0 : self.lens[0], 1:2].clone()
        t_e = t_e.requires_grad_()
        u_e = self.neural_net(torch.cat([x_e, t_e], dim = 1))
        u_t = grad(u_e, t_e)
        u_x = grad(u_e, x_e)
        u_xx = grad(u_x, x_e)
        equation = u_t + u_e * u_x - (0.01 / np.pi) * u_xx 
        
        # initial condition
        x_i = X[self.lens[0]: self.lens[1], 0:1]
        u_i = self.neural_net(X[self.lens[0]: self.lens[1], :])
        initial_value = - torch.sin(np.pi * x_i)
        initial = u_i - initial_value
        
        # boundary condition
        Dvalue = 0 
        ## left
        u_b1 = self.neural_net(X[self.lens[1]: self.lens[2], :])
        boundary1 = u_b1 - Dvalue
        ## right
        u_b2 = self.neural_net(X[self.lens[2]: self.lens[3], :])
        boundary2 = u_b2 - Dvalue
        
        return equation, initial, boundary1, boundary2

    # loss function
    def loss(self,X_train):
        # governing equation, initial conditon, boundary condition
        equation, initial, boundary1, boundary2 = self.he_net(X_train)
        
        # total loss
        loss_e = torch.mean(torch.square(equation))
        loss_i = torch.mean(torch.square(initial))
        loss_b1 = torch.mean(torch.square(boundary1))
        loss_b2 = torch.mean(torch.square(boundary2))
        loss_all = loss_e + self.initial_weight * loss_i + self.boundary_weight * loss_b1 + self.boundary_weight * loss_b2

        return loss_all, loss_e, loss_i, loss_b1, loss_b2

    # Predict the value of the density u at the corresponding point of X
    def predict(self, X):
        u_pred = self.neural_net(X)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred 
        
    # Predict the value of the density u of a channel at the corresponding point of X
    def predict_channel(self, X, channel):
        u_pred = self.neural_net_channel(X, channel)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred

# main
if __name__ == "__main__":
    seeds = [19]
    for seed in seeds: 
        print("***** seed = %d *****" % seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
       
        # parameter settings
        ## equation related
        lb_x = -1 
        ub_x = 1 
        lb_t = 0
        ub_t = 1
        
        ## Neural network related
        train = True
        name = "burgers_512-32"
        N0 = 200 # number of training points corresponding to initial condition
        Nb = 100 # The number of training points on each edge
        Nf = 30000 # number of training points corresponding to governing equations
        epochs = 10000 # Number of Adam optimizer iterations
        Layers = [2,512,256,128,64,32,1] # fully connected neural network structure
        learning_rate = 0.005  
        patience = max(10, epochs / 10) # patience for decaying learning rate
        initial_weight = 1 # initial weight
        boundary_weight = 1 # boundary weight
        weight_decay = 0.0001
        
        ## draw related
        align = True
        lb_loss = 1e-3
        ub_loss = 1e21
        lb_u = -1.2
        ub_u = 1.2
        
        # Auxiliary variables
        name = name + ("_%d" % epochs)
        print("\n\n***** name = %s *****" % name)
        print("seed = %d" % seed)
        output_path = path + ('./output')
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('./output/%s/' % name)
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('./output/%s/train_%d/' % (name, seed))
        if not os.path.exists(output_path): os.mkdir(output_path)


        if train:
            # generate train set
            print("Loading data")
            X_train, u_truth = load_data(N0, Nb, Nf, lb_x, ub_x, lb_t, ub_t)
            lb_X = X_train.min(0) 
            ub_X = X_train.max(0) 
            lb_X = torch.from_numpy(lb_X).float().to(device)
            ub_X = torch.from_numpy(ub_X).float().to(device)
            X_train = torch.from_numpy(X_train).float().to(device)

            # Declare neural network instance
            model = BsPINN(Layers, N0, Nb, Nf, initial_weight, boundary_weight, lb_X, ub_X)
            model = nn.DataParallel(model)
            model = model.module
            model.to(device)
            print(model)
            
            # Adam optimizer
            optimizer = optim.Adam(model.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, verbose=True, min_lr=1e-6) 

            # train
            start = time.time()
            ## adam
            loss_list = []
            min_loss = 9999999
            print("Start training!")
            for it in range(epochs):
                # Optimizer training
                Loss, loss_e, loss_i, loss_b1, loss_b2 = model.loss(X_train)
                optimizer.zero_grad(set_to_none=True)
                Loss.backward() 
                optimizer.step()

                # decay learning rate
                scheduler.step(Loss)

                # save train loss
                loss_val = Loss.cpu().detach().numpy()
                loss_e_val = loss_e.cpu().detach().numpy()
                loss_i_val = loss_i.cpu().detach().numpy()
                loss_b1_val = loss_b1.cpu().detach().numpy()
                loss_b2_val = loss_b2.cpu().detach().numpy()
                loss_list.append(loss_val)
                if loss_val < min_loss: # save the model with the minimum train loss
                    torch.save(model, output_path + 'network.pkl') 
                    min_loss = loss_val

                # print intermediate results
                if (it + 1) % (epochs/20) == 0:
                    print("It = %d, loss = " % (it + 1), loss_val, ", finish: %d%%" % ((it + 1) / epochs * 100))
                
            # following processing
            end = time.time()
            train_time = end - start
            loss_list = np.array(loss_list).flatten()
            np.savetxt(output_path + "loss.txt", loss_list, fmt="%s",delimiter=' ')
            params = get_parameter_number(model) 
            min_loss = np.min(loss_list)
            final_lr = optimizer.param_groups[0]['lr']
            print("time = %.2fs" % train_time)
            print("params = %d" % params)
            print("min_loss = %.8f" % min_loss)
        else:
            X_train, u_truth = load_data(N0, Nb, Nf, lb_x, ub_x, lb_t, ub_t)

        # save loss curve
        plt.rcParams['font.size'] = 20
        loss_list = np.loadtxt(output_path + "loss.txt", dtype = float, delimiter=' ')
        plt.figure()
        plt.semilogy(loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if align:
            plt.ylim(lb_loss, ub_loss) 
            plt.savefig(output_path + 'loss_aligned.pdf', format="pdf", dpi=100, bbox_inches="tight")
        else:
            plt.savefig(output_path + 'loss.pdf', format="pdf", dpi=100, bbox_inches="tight")

        # error assessment
        dimx = 256
        dimt = 100
        X = np.linspace(lb_x, ub_x, dimx)
        T = np.linspace(lb_t, ub_t, dimt, endpoint = False)
        X, T = np.meshgrid(X, T)
        X = X.flatten().reshape(dimx * dimt, 1)
        T = T.flatten().reshape(dimx * dimt, 1)
        points = np.c_[X, T] # N * 2
        X_pred = points
        X_pred = torch.from_numpy(X_pred).float().to(device_cpu)
        model2 = torch.load(output_path + 'network.pkl', map_location=device_cpu) 
        u_pred = model2.predict(X_pred).flatten()
        u_error = np.linalg.norm((u_truth - u_pred),ord=2) / (np.linalg.norm(u_truth, ord=2))
        print("u_error = %.8f when drawing." % u_error)

        # draw the image of predicted solution
        plt.rcParams['font.size'] = 30
        fig, ax = plt.subplots()
        if align:
                levels = np.arange(lb_u, ub_u + 1e-8, (ub_u - lb_u) / 100)
        else:
            levels = np.arange(min(u_pred) - abs(max(u_pred) - min(u_pred)) / 10, max(u_pred) + abs(max(u_pred) - min(u_pred)) / 10, (max(u_pred) - min(u_pred)) / 100) 
        cs = ax.contourf(X.reshape(dimt,dimx), T.reshape(dimt,dimx), u_pred.reshape(dimt,dimx), levels,cmap=plt.get_cmap('Spectral'))
        cbar = fig.colorbar(cs)
        plt.xticks([-1,-0.5,0,0.5,1])
        plt.yticks([0.00,0.25,0.50,0.75,1.00])
        plt.xlabel('$x$')
        plt.ylabel('$t$')
        plt.title('$u$(BsPINN)')
        if align:
            plt.savefig(output_path + "u_pred_aligned.png", format="png", dpi=150, bbox_inches="tight")
        else:
            plt.savefig(output_path + "u_pred.png", format="png", dpi=150, bbox_inches="tight")
            
        # draw the image of exact solution
        fig, ax = plt.subplots()
        if align:
                levels = np.arange(lb_u, ub_u + 1e-8, (ub_u - lb_u) / 100)
        else:
            levels = np.arange(min(u_truth) - abs(max(u_truth) - min(u_truth)) / 10, max(u_truth) + abs(max(u_truth) - min(u_truth)) / 10, (max(u_truth) - min(u_truth)) / 100) 
        cs = ax.contourf(X.reshape(dimt,dimx), T.reshape(dimt,dimx), u_truth.reshape(dimt,dimx), levels,cmap=plt.get_cmap('Spectral'))
        cbar = fig.colorbar(cs)
        plt.xticks([-1,-0.5,0,0.5,1])
        plt.yticks([0.00,0.25,0.50,0.75,1.00])
        plt.xlabel('$x$')
        plt.ylabel('$t$')
        plt.title('$u$(truth)')
        if align:
            plt.savefig(output_path + "u_truth_aligned.png", format="png", dpi=150, bbox_inches="tight")
        else:
            plt.savefig(output_path + "u_truth.png", format="png", dpi=150, bbox_inches="tight")
        
        # draw the image of a channel of predicted solution
        # dimx = 500
        # X = np.linspace(lb_x, ub_x, dimx)
        # T = np.linspace(lb_t, ub_t, dimt, endpoint = False)
        # X, T = np.meshgrid(X, T)
        # X = X.flatten().reshape(dimx * dimt, 1)
        # T = T.flatten().reshape(dimx * dimt, 1)
        # points = np.c_[X, T] # N * 2
        # X_pred = points
        # X_pred = torch.from_numpy(X_pred).float().to(device)
        # model2 = torch.load(output_path + 'network.pkl', map_location=device_cpu)   
        # for channel in range(int(pow(2,len(Layers) - 3))):
        #     print("channel=%d" % channel)
        #     u_pred = model2.predict_channel(X_pred, channel).flatten()
        #     fig, ax = plt.subplots()
        #     if align:
        #             levels = np.arange(lb_u, ub_u + 1e-8, (ub_u - lb_u) / 100)
        #     else:
        #         levels = np.arange(min(u_pred) - abs(max(u_pred) - min(u_pred)) / 10, max(u_pred) + abs(max(u_pred) - min(u_pred)) / 10, (max(u_pred) - min(u_pred)) / 100) 
        #     cs = ax.contourf(X.reshape(dimt,dimx), T.reshape(dimt,dimx), u_pred.reshape(dimt,dimx), levels,cmap=plt.get_cmap('Spectral'))
        #     cbar = fig.colorbar(cs)
        #     plt.xlabel('$x$')
        #     plt.ylabel('$x$')
        #     plt.title('channel%d' % channel)
        #     plt.savefig(output_path + "per_channel/%d.png" % channel)