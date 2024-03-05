import os
import numpy as np
from matplotlib import pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
import math
from idaes.surrogate.pysmo.sampling import HammersleySampling
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from torch.nn.parameter import Parameter

# System settings
torch.set_default_dtype(torch.float)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Device configuration
path = os.path.dirname(__file__) + "/"
torch.backends.cudnn.benchmark = True
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 20

# exact solution
def solution(x,y):
    return np.sin(kappa * x) * np.sin(kappa * y)

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
def load_data(num_domain, num_boundary, nx_test, lb_geom, ub_geom):
    # governing equation
    delta = 0.01
    X_train = np.random.uniform([delta, delta], [1 - delta, 1 - delta], (num_domain, 2))
    # boundary condition
    points = np.linspace([lb_geom,lb_geom],[ub_geom,lb_geom],num_boundary,endpoint=False) # 下
    X_train = np.r_[X_train, points]
    points = np.linspace([ub_geom,lb_geom],[ub_geom,ub_geom],num_boundary,endpoint=False) # 右
    X_train = np.r_[X_train, points]
    points = np.linspace([lb_geom,lb_geom],[lb_geom,ub_geom],num_boundary,endpoint=False) # 左
    X_train = np.r_[X_train, points]
    points = np.linspace([lb_geom,ub_geom],[ub_geom,ub_geom],num_boundary,endpoint=False) # 上
    X_train = np.r_[X_train, points]

    # target
    Y_train = np.zeros((num_domain + 4 * num_boundary, 1))

    # test set
    X = np.linspace(lb_geom, ub_geom, nx_test)
    Y = np.linspace(lb_geom, ub_geom, nx_test)
    X,Y = np.meshgrid(X,Y)
    X = X.flatten()
    Y = Y.flatten()
    X_test = np.c_[X,Y]
    Y_test = solution(X,Y)
    Y_test = np.reshape(Y_test, (Y_test.shape[0], 1))

    return X_train, Y_train, X_test, Y_test

class BsPINN(nn.Module):
    def __init__(self, Layers, k0, num_domain, boundary_weight, lb_X, ub_X):
        super(BsPINN, self).__init__()
        # initialize parameters
        self.mode = "train"
        self.k0 = k0
        self.lb_X = lb_X
        self.ub_X = ub_X
        self.Layers = Layers
        self.num_domain = num_domain
        self.boundary_weight = boundary_weight
        self.loss_function = nn.MSELoss(reduction ='mean')
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
        out = temp[num_Layers - 3][0]
        for i in range(1, len(temp[num_Layers - 3])):
            out = torch.concat([out,temp[num_Layers - 3][i]],1)
        Y = torch.add(torch.matmul(out, self.w_last),self. b_last)
        return Y
    
    # binary structured neural network for a channel
    def neural_net_channel(self, X, channel):
        # Data preprocessing
        if torch.is_tensor(X) != True:         
            X = torch.from_numpy(X)
        ub = torch.from_numpy(self.ub_X).float().to(device)
        lb = torch.from_numpy(self.lb_X).float().to(device)  
        X = 2.0 * (X - lb) / (ub - lb) - 1.0
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

    # binary structured neural network for a hidden layer
    def neural_net_layer(self, X, num_layer):
        # Data preprocessing
        if torch.is_tensor(X) != True:         
            X = torch.from_numpy(X)
        ub = torch.from_numpy(self.ub_X).float().to(device)
        lb = torch.from_numpy(self.lb_X).float().to(device)  
        X = 2.0 * (X - lb) / (ub - lb) - 1.0
        H = X.float()

        # the first hidden layer
        num_Layers = len(self.Layers)
        l_out = torch.sin(torch.add(torch.matmul(H, self.weights[0][0]), self.biases[0][0]))
        temp = [[] for i in range(num_Layers - 2)]# save the outputs of each hidden layer.
        temp[0].append(l_out)
        # 之后的隐藏层
        for l in range(1,num_Layers-2):
            for i in range(int(pow(2,l))):
                W = self.weights[l][i]
                b = self.biases[l][i]
                l_out = torch.sin(torch.add(torch.matmul(temp[l-1][int(i/2)], W), b))
                temp[l].append(l_out)
        # the following hidden layer
        out_list = []
        dim = self.Layers[num_layer + 1]
        for i in range(len(temp[num_layer])):
            out_channel = torch.add(torch.matmul(temp[num_layer][i], self.w_last[i * dim : (i + 1) * dim , :]), self.b_last)
            out_list.append(out_channel)
        Y = torch.concatenate(out_list, dim=1)
        return Y

    # PDE part
    def he_net(self, x, y):
        # governing equation
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

        # boundary condition
        x_b = x[self.num_domain:].clone()
        y_b = y[self.num_domain:].clone()
        u_b = self.neural_net(torch.cat([x_b, y_b], dim = 1))
        boundary = u_b - 0

        # total displacement
        u = torch.cat([u_e, u_b], dim = 0)

        return u, equation, boundary

    # loss function
    def loss(self,X_train,Y_train):
        # governing equation, initial conditon, boundary condition
        x = X_train[:,0:1]
        y = X_train[:,1:2]
        _, equation, boundary = self.he_net(x, y)
        
        # total loss
        loss_e = self.loss_function(equation, Y_train[0:self.num_domain])
        loss_b = self.loss_function(boundary, Y_train[self.num_domain:])
        loss_all = loss_e + self.boundary_weight * loss_b

        return loss_all
    
    # calculate relative error
    def rel_error(self):
        u_pred = self.neural_net(X_test).cpu().detach().numpy()
        u_truth = Y_test.cpu().detach().numpy()
        u_error = np.linalg.norm((u_truth - u_pred), ord=2) / (np.linalg.norm(u_truth, ord=2))
        return u_error
    
    # Predict the value of u at the corresponding point of X
    def predict(self, X):
        u_pred = self.neural_net(X)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred
    
    # Predict the value of u of a channel at the corresponding point of X
    def predict_channel(self, X, channel):
        u_pred = self.neural_net_channel(X, channel)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred
    
    # Predict the value of u of a hidden layer at the corresponding point of X
    def predict_layer(self, X, num_layer):
        u_pred = self.neural_net_layer(X, num_layer)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred

# main
if __name__ == "__main__":
    seeds = [16]
    for seed in seeds:
        print("***** seed = %d *****" % seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # parameter settings
        ## equation related
        train = True
        name = "helmholtz_2d_256-16"
        precision_train = 20 # Training precision, the number of training points on a complete cycle of wave
        precision_test = 10 # Test precision, the number of test points on a complete cycle of wave
        epochs = 80000 # Number of Adam optimizer iterations
        Layers = [2, 256, 128, 64, 32, 16, 1] # binary structured neural network structure
        learning_rate = 0.001
        boundary_weight = 100
        patience = max(10, epochs/10)
        weight_decay = 0
        
        ## equation related
        lb_geom = 0
        ub_geom = 1
        kappa = 8 * np.pi # Number of complete cycles of waves on a line segment of length 1
        
        ## draw related
        align = False 
        lb_loss = 1e-2
        ub_loss = 1e9
        lb_u = -1.45
        ub_u = 1.45
        
        # Auxiliary variables
        name = name + ("_%d" % epochs)
        print("\n\n***** name = %s *****" % name)
        print("seed = %d" % seed)
        omiga = kappa / (2 * np.pi)
        wave_len = 1 / omiga 
        hx_train = wave_len / precision_train 
        nx_train = int((ub_geom - lb_geom) / hx_train) + 1 
        hx_test = wave_len / precision_test
        nx_test = int((ub_geom - lb_geom) / hx_test) + 1  
        output_path = path + ('./output')
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('./output/%s' % name)
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('./output/%s/train_%d/' % (name, seed))
        if not os.path.exists(output_path): os.mkdir(output_path)

        if train:
            # generate train set
            print("Loading data.")
            num_domain = int(nx_train ** 2) 
            num_boundary = nx_train - 1 
            X_train, Y_train, X_test, Y_test = load_data(num_domain, num_boundary, nx_test, lb_geom, ub_geom)
            lb_X = X_train.min(0)
            ub_X = X_train.max(0)
            X_train = torch.from_numpy(X_train).float().to(device)
            Y_train = torch.from_numpy(Y_train).float().to(device)
            X_test = torch.from_numpy(X_test).float().to(device)
            Y_test = torch.from_numpy(Y_test).float().to(device)

            # Declare neural network instance
            model = BsPINN(Layers, kappa, num_domain, boundary_weight, lb_X, ub_X)
            model = nn.DataParallel(model)
            model = model.module
            model.to(device)
            print(model) 
            
            # Adam optimizer
            optimizer = optim.Adam(model.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=patience, verbose=True, min_lr=1e-6)

            # train
            start = time.time()
            loss_list = []
            min_loss = 9999999
            print("Start training")
            for it in range(epochs):
                # Optimizer training
                loss = model.loss(X_train, Y_train) 
                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

                # decay learning rate
                scheduler.step(loss)

                # save loss
                loss_val = loss.cpu().detach().numpy()
                loss_list.append(loss_val)

                # save the model with the minimum train loss
                if loss_val < min_loss:
                    torch.save(model, output_path + 'network.pkl')
                    min_loss = loss_val

                # print intermediate results
                if (it + 1) % (epochs/20) == 0:
                    print("It = %d, loss = " % (it + 1), loss_val,", u_error = ", model.rel_error(), ", finish: %d%%" % ((it + 1) / epochs * 100))
            end = time.time()
            print("Total train time: %.2fs" % (end - start))
            loss_list = np.array(loss_list).flatten()
            np.savetxt(output_path + "loss.txt", loss_list, fmt="%s",delimiter=' ')
            min_loss = np.min(loss_list)
            print("Min train loss: %.8f" % min_loss)
            params = get_parameter_number(model) 
            print("Total number of parameters: %d" % params)
        torch.cuda.empty_cache() 
        
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


        # error assessment
        dim = 500
        X = np.linspace(lb_geom, ub_geom, dim)
        Y = np.linspace(lb_geom, ub_geom, dim)
        X, Y = np.meshgrid(X, Y)
        u_truth = solution(X.flatten(),Y.flatten())
        X = X.flatten().reshape(dim*dim,1)
        Y = Y.flatten().reshape(dim*dim,1)
        X_pred = np.c_[X,Y]
        X_pred = torch.from_numpy(X_pred).float().to(device)
        model2 = torch.load(output_path + 'network.pkl', map_location=device) 
        u_pred = model2.predict(X_pred).flatten()
        u_mae = np.mean(np.abs(u_truth - u_pred))
        print("u_mae = %.8f" % u_mae)
        u_error = np.linalg.norm((u_truth - u_pred),ord=2) / (np.linalg.norm(u_truth, ord=2))
        print("u_error = %.8f" % u_error)
        np.savetxt(output_path + "u_pred.txt", u_pred, fmt="%s", delimiter=' ')

        # draw the image of predicted solution
        fig, ax = plt.subplots()
        if align:
                levels = np.arange(lb_u, ub_u + 1e-8, (ub_u - lb_u) / 100)
        else:
            levels = np.arange(min(u_pred) - abs(max(u_pred) - min(u_pred)) / 10, max(u_pred) + abs(max(u_pred) - min(u_pred)) / 10, (max(u_pred) - min(u_pred)) / 100) 
        cs = ax.contourf(X.reshape(dim,dim), Y.reshape(dim,dim), u_pred.reshape(dim,dim), levels,cmap=plt.get_cmap('Spectral'))
        cbar = fig.colorbar(cs)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('$u$(BsPINN)')
        if align:
            plt.savefig(output_path + "u_pred_aligned.png", format="png", dpi=100, bbox_inches="tight")
        else:
            plt.savefig(output_path + "u_pred.png", format="png", dpi=100, bbox_inches="tight")
            
        # draw the image of exact solution
        fig, ax = plt.subplots()
        if align:
                levels = np.arange(lb_u, ub_u + 1e-8, (ub_u - lb_u) / 100)
        else:
            levels = np.arange(min(u_truth) - abs(max(u_truth) - min(u_truth)) / 10, max(u_truth) + abs(max(u_truth) - min(u_truth)) / 10, (max(u_truth) - min(u_truth)) / 100) 
        cs = ax.contourf(X.reshape(dim,dim), Y.reshape(dim,dim), u_truth.reshape(dim,dim), levels,cmap=plt.get_cmap('Spectral'))
        cbar = fig.colorbar(cs)
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('$u$(truth)')
        if align:
            plt.savefig(output_path + "u_truth_aligned.png", format="png", dpi=100, bbox_inches="tight")
        else:
            plt.savefig(output_path + "u_truth.png", format="png", dpi=100, bbox_inches="tight")
        
        # # draw the image of each hidden layer of predicted solution 
        # dim = 500
        # X = np.linspace(lb_geom, ub_geom, dim)
        # Y = np.linspace(lb_geom, ub_geom, dim)
        # X, Y = np.meshgrid(X, Y)
        # X = X.flatten().reshape(dim*dim,1)
        # Y = Y.flatten().reshape(dim*dim,1)
        # X_pred = np.c_[X,Y]
        # X_pred = torch.from_numpy(X_pred).float().to(device)
        # model2 = torch.load(output_path + 'network.pkl') 
        # for channel in range(int(pow(2,len(Layers) - 3))):
        #     print("channel=%d" % channel)
        #     u_pred = model2.predict_channel(X_pred, channel).flatten()
        #     np.savetxt(output_path + "per_channel_bipinn/%d.txt" % channel, u_pred, fmt="%s", delimiter=' ')
        #     # 画每个通道的图像
        #     # fig, ax = plt.subplots()
        #     # if align:
        #     #     levels = np.arange(lb_u, ub_u, (ub_u - lb_u) / 100)
        #     # else:
        #     #     levels = np.arange(min(u_pred) - abs(max(u_pred) - min(u_pred)) / 10, max(u_pred) + abs(max(u_pred) - min(u_pred)) / 10, (max(u_pred) - min(u_pred)) / 100) 
        #     # cs = ax.contourf(X.reshape(dim,dim), Y.reshape(dim,dim), u_pred.reshape(dim,dim), levels,cmap=plt.get_cmap('Spectral'))
        #     # cbar = fig.colorbar(cs)
        #     # plt.xlabel('this is x')
        #     # plt.ylabel('this is y')
        #     # plt.title('u_pred_channel%d' % channel)
        #     # if align:
        #     #     img_path = output_path + "per_channel_aligned/"
        #     #     if not os.path.exists(img_path): os.mkdir(img_path)
        #     #     plt.savefig(img_path + "u_pred_channel_%d.png" % channel)
        #     # else:
        #     #     img_path = output_path + "per_channel/"
        #     #     if not os.path.exists(img_path): os.mkdir(img_path)
        #     #     plt.savefig(img_path + "u_pred_channel_%d.png" % channel)
        
        # # draw the image of each channel of predicted solution 
        # dim = 500
        # X = np.linspace(lb_geom, ub_geom, dim)
        # Y = np.linspace(lb_geom, ub_geom, dim)
        # X, Y = np.meshgrid(X, Y)
        # X = X.flatten().reshape(dim*dim,1)
        # Y = Y.flatten().reshape(dim*dim,1)
        # X_pred = np.c_[X,Y]
        # X_pred = torch.from_numpy(X_pred).float().to(device)
        # model2 = torch.load(output_path + 'network.pkl') 
        # for num_layer in range(len(Layers) - 2):
        #     print("layer=%d" % num_layer)
        #     u_layer = model2.predict_layer(X_pred, num_layer).flatten()
        #     print("Layer %d: max = %.8f, min = %.8f" % (num_layer, np.max(u_layer), np.min(u_layer)))
        #     np.savetxt(output_path + "per_layer/%d.txt" % num_layer, u_layer, fmt="%s", delimiter=' ')