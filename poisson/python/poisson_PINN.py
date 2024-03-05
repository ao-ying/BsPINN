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

# System settings
torch.set_default_dtype(torch.float)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Device configuration
device_cpu = torch.device('cpu')
path = os.path.dirname(__file__) + "/"
torch.backends.cudnn.benchmark = True
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 30

# exact solution
def solution(x):
    if len(x.shape) > 1: # x is a matrix with dimension n * d
        x_sum = np.sum(x, axis=1)
        x_mean = np.mean(x, axis=1)
    else :
        x_sum = np.sum(x) # x is a vector with dimension d
        x_mean = np.mean(x)
    ustar = x_mean ** 2 + np.sin(coe * x_sum) # x_mean ** 2
    return ustar

# partial derivative
def grad(f,x):
    ret = torch.autograd.grad(f, x, torch.ones_like(f).to(device), retain_graph=True, create_graph=True)[0]
    return ret

# output model parameter information
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.params)
    trainable_num = sum(p.numel() for p in net.params if p.requires_grad)
    return trainable_num
    
# reset learning rate
def reset_lr(optimizer, lr):
    for params in optimizer.param_groups: 
            params['lr'] = lr

# generate dataset
def load_data(lb_geom, ub_geom, num_sample, num_boundary, dim_x):
    # governing equation
    X_train = np.random.uniform(lb_geom, ub_geom, (num_sample, dim_x))
    # boundary condition
    num_per_dim = int(num_boundary / dim_x)
    tool = np.zeros(num_per_dim)
    tool[:int(num_per_dim/2)] = ub_geom
    tool[int(num_per_dim/2):] = lb_geom
    for i in range(dim_x):
        boundary_points = np.random.uniform(lb_geom, ub_geom, (num_per_dim, dim_x))
        boundary_points[:, i] = tool
        X_train = np.r_[X_train, boundary_points]
    
    return X_train

# Residual block
class Residual(nn.Module):
    def __init__(self, Layers):
        super().__init__()
        # initialize data
        self.data_dim = Layers[0] # width of the fully connected neural network
        self.num_fc = len(Layers) # number of hidden layer
        
        # initialize parameters
        self.params = []
        self.weights = []
        self.biases = []
        for l in range(0,self.num_fc):
            tempw = torch.zeros(self.data_dim, self.data_dim).to(device)
            w = Parameter(tempw, requires_grad=True)
            nn.init.xavier_uniform_(w, gain=1) 
            tempb = torch.zeros(1, self.data_dim).to(device)
            b = Parameter(tempb, requires_grad=True)
            self.weights.append(w)
            self.biases.append(b) 
            self.params.append(w)
            self.params.append(b)
        
    def forward(self, X):
        H = X
        for l in range(0, self.num_fc):
            W = self.weights[l]
            b = self.biases[l]
            H = torch.add(torch.matmul(H, W), b)
            if l != self.num_fc - 1: # the last hidden layer doesn't have activation function
                H = torch.sin(H)
        Y = torch.sin(torch.add(H, X))
        return Y

class PINN(nn.Module):
    def __init__(self, Layers, num_sample, num_boundary, boundary_weight, lb_X, ub_X, num_res, dim_x, dim_y, coe):
        super(PINN, self).__init__()
        # initialize parameters# 初始化参数
        self.lb_X = lb_X
        self.ub_X = ub_X
        self.Layers = Layers
        self.num_sample = num_sample
        self.boundary_weight = boundary_weight
        self.loss_function = nn.MSELoss(reduction ='mean')
        self.num_res = num_res
        self.params = []
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_res_fc = Layers[0] # width of binary structured neural network in residual block
        self.num_boundary = num_boundary
        self.coe = coe
        # initialize tne first fully connected layer
        self.fc_first = nn.Linear(dim_x, self.dim_res_fc)
        nn.init.xavier_uniform_(self.fc_first.weight, gain=1)
        nn.init.zeros_(self.fc_first.bias)
        self.params.append(self.fc_first.weight)
        self.params.append(self.fc_first.bias)
        # initialize the residual blocks
        self.res_blocks = nn.ModuleList([Residual(Layers) for i in range(self.num_res)])
        for i in range(self.num_res):
            self.params.extend(self.res_blocks[i].params)
        # initialize tne last fully connected layer
        self.fc_last = nn.Linear(self.dim_res_fc, self.dim_y)
        nn.init.xavier_uniform_(self.fc_last.weight, gain=1)
        nn.init.zeros_(self.fc_last.bias)
        self.params.append(self.fc_last.weight)
        self.params.append(self.fc_last.bias)
    
    # neural network part
    def neural_net(self, X):
        # Data preprocessing
        X = 2.0 * (X - self.lb_X) / (self.ub_X - self.lb_X) - 1.0
        H = X.float()
        # ResNet part
        H = self.fc_first(H)
        for i in range(self.num_res):
            H = self.res_blocks[i](H)
        Y = self.fc_last(H)

        return Y

    # PDE part
    def he_net(self, X):
        # governing equation
        X_e = [0 for i in range(self.dim_x)]
        for i in range(self.dim_x):
            X_e[i] = X[0:self.num_sample, i : i + 1].clone()
            X_e[i] = X_e[i].requires_grad_()
        u_e = self.neural_net(torch.cat(X_e, dim = 1))
        dudx = [grad(u_e, X_e[i]) for i in range(self.dim_x)]
        dudx2 = [grad(dudx[i], X_e[i]) for i in range(self.dim_x)]
        dudx2 = torch.cat(dudx2, dim=1) # self.num_sample * dim_x
        Laplace_u = torch.sum(dudx2, dim=1, keepdim=True)  # self.num_sample * 1
        sum_x_e = torch.sum(X[0:self.num_sample, :], dim=1, keepdim=True) # self.num_sample * 1
        f = self.dim_x * (self.coe ** 2) * torch.sin(self.coe * sum_x_e) - 2 / self.dim_x # right hand side
        equation = - Laplace_u - f

        # boundary condition
        X_b = [0 for i in range(self.dim_x)]
        for i in range(self.dim_x):
            X_b[i] = X[self.num_sample:, i : i + 1].clone()
        u_b = self.neural_net(torch.cat(X_b, dim = 1)) # self.num_boundary * 1
        sum_x_b = torch.sum(X[self.num_sample:, :], dim=1, keepdim=True) # self.num_boundary * 1
        mean_x_b = sum_x_b / self.dim_x
        Dvalue = torch.pow(mean_x_b, 2) + torch.sin(self.coe * sum_x_b) 
        boundary = u_b - Dvalue

        # total displacement
        u = torch.cat([u_e, u_b], dim = 0) # (self.num_sample + self.num_boundary) * 1

        return u, equation, boundary

    # loss function
    def loss(self,X_train):
        # governing equation, initial conditon, boundary condition
        _, equation, boundary = self.he_net(X_train)
        
        # total loss
        loss_e = torch.mean(torch.square(equation))
        loss_b = torch.mean(torch.square(boundary))
        loss_all = loss_e + self.boundary_weight * loss_b

        return loss_all
    
    # Predict the value of u at the corresponding point of X
    def predict(self, X):
        u_pred = self.neural_net(X)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred 
    
    def set_device(self, device):
        self.lb_X = self.lb_X.to(device)
        self.ub_X = self.ub_X.to(device)

# main
if __name__ == "__main__":
    seeds = [29]
    for seed in seeds: 
        np.random.seed(seed)
        torch.manual_seed(seed)
        # parameter settings
        ## equation related
        lb_geom = -1 
        ub_geom = 1 
        dim_x = 10 
        dim_y = 1
        coe = 0.6 * np.pi
        
        ## Neural network related
        name = "poisson_fc_6*256"
        num_sample = 4000
        num_boundary = 2000 
        epochs = 10000 # Number of Adam optimizer iterations
        Layers = [256, 256, 256, 256, 256, 256] # fully connected neural network structure
        learning_rate = 0.001
        boundary_weight = 1 
        num_res = 2 # number of residual blocks
        h = int(1000) 
        r = 0.95 
        weight_decay = 0.001 
        
        ## draw related
        train = True
        align = False
        lb_loss = 1e-3
        ub_loss = 1e11
        lb_error = 1e-2
        ub_error = 1e1
        lb_u = -1.1
        ub_u = 1.1
        
        # Auxiliary variables
        name = name + ("_%d" % epochs)
        print("\n\n***** name = %s *****" % name)
        print("seed = %d" % seed)
        output_path = path + ('./output')
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('./output/%s' % name)
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('./output/%s/train_%d/' % (name, seed))
        if not os.path.exists(output_path): os.mkdir(output_path)

        if train:
            # generate train set
            print("Loading data")
            X_train = load_data(lb_geom, ub_geom, num_sample, num_boundary, dim_x)
            lb_X = X_train.min(0) 
            ub_X = X_train.max(0) 
            lb_X = torch.from_numpy(lb_X).float().to(device)
            ub_X = torch.from_numpy(ub_X).float().to(device)
            X_train = torch.from_numpy(X_train).float().to(device)

            # Declare neural network instance
            model = PINN(Layers, num_sample, num_boundary, boundary_weight, lb_X, ub_X, num_res, dim_x, dim_y, coe)
            model = nn.DataParallel(model)
            model = model.module
            model.to(device)
            print(model)
            
            # Adam optimizer
            optimizer = optim.Adam(model.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)

            # train
            start = time.time()
            ## adam
            loss_list = []
            error_list = [] # 保存在测试集上的平均相对误差值
            min_loss = 9999999
            print("Start training!")
            for it in range(epochs):
                # Optimizer training
                Loss = model.loss(X_train)
                optimizer.zero_grad(set_to_none=True)
                Loss.backward() 
                optimizer.step()

                # decay learning rate
                if (it + 1) % h == 0: 
                    learning_rate *= r
                    reset_lr(optimizer, learning_rate)

                # save loss value and model with the minimum train loss
                loss_val = Loss.cpu().detach().numpy()
                loss_list.append(loss_val)
                if loss_val < min_loss: 
                        torch.save(model, output_path + 'network.pkl') 
                        min_loss = loss_val

                # print intermediate results
                if (it + 1) % (epochs/20) == 0:
                    print("It = %d, loss = " % (it + 1), loss_val, ", finish: %d%%" % ((it + 1) / epochs * 100))
                
            # following processing
            end = time.time()
            train_time = end - start
            loss_list = np.array(loss_list).flatten()
            error_list = np.array(error_list).flatten()
            min_loss = np.min(loss_list)
            np.savetxt(output_path + "loss.txt", loss_list, fmt="%s",delimiter=' ')
            params = get_parameter_number(model)
            print("time = %.2fs" % train_time)
            print("params = %d" % params)
            print("min_loss = %.8f" % min_loss)

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
        dim_gauss = 4
        points1 = np.loadtxt(path + "../data/gauss_points_%d_1.txt" % dim_gauss, dtype = float, delimiter=' ')
        points2 = np.loadtxt(path + "../data/gauss_points_%d_2.txt" % dim_gauss, dtype = float, delimiter=' ')
        points = np.r_[points1, points2]
        weights = np.loadtxt(path + "../data/gauss_weights_%d.txt" % dim_gauss, dtype = float, delimiter=' ')
        X_pred = torch.from_numpy(points).float().to(device_cpu)
        model2 = torch.load(output_path + 'network.pkl', map_location=device_cpu) 
        u_pred = model2.predict(X_pred).flatten()
        u_truth = solution(points)
        u_error = np.sum(weights * np.square(u_pred - u_truth)) / np.sum((weights * np.square(u_truth)))
        print("u_error = %.10f" % u_error)
        np.savetxt(output_path + "u_truth.txt", u_truth, fmt="%s", delimiter=' ')
        np.savetxt(output_path + "u_pred.txt", u_pred, fmt="%s", delimiter=' ')
        
        # prepare data for drawing
        dim = 500
        X1 = np.linspace(lb_geom, ub_geom, dim)
        X2 = np.linspace(lb_geom, ub_geom, dim)
        X1, X2 = np.meshgrid(X1, X2)
        X1 = X1.flatten().reshape(dim*dim,1)
        X2 = X2.flatten().reshape(dim*dim,1)
        points = np.c_[X1, X2] # N * 2
        tool = np.zeros((X1.shape[0], dim_x - 2)) 
        points = np.c_[points, tool] 
        u_truth = solution(points)
        X_pred = points
        X_pred = torch.from_numpy(X_pred).float().to(device_cpu)
        model2 = torch.load(output_path + 'network.pkl', map_location=device_cpu) 
        if train: model2.set_device(device_cpu)
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
        cs = ax.contourf(X1.reshape(dim,dim), X2.reshape(dim,dim), u_pred.reshape(dim,dim), levels,cmap=plt.get_cmap('Spectral'))
        cbar = fig.colorbar(cs)
        plt.xticks([-1,-0.5,0,0.5,1])
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('$u$(PINN)')
        if align:
            plt.savefig(output_path + "u_pred_aligned.png", format="png", dpi=100, bbox_inches="tight")
        else:
            plt.savefig(output_path + "u_pred.png", format="png", dpi=100, bbox_inches="tight")
        
        # draw the image of exact solution
        plt.rcParams['font.size'] = 30
        fig, ax = plt.subplots()
        if align:
                levels = np.arange(lb_u, ub_u + 1e-8, (ub_u - lb_u) / 100)
        else:
            levels = np.arange(min(u_truth) - abs(max(u_truth) - min(u_truth)) / 10, max(u_truth) + abs(max(u_truth) - min(u_truth)) / 10, (max(u_truth) - min(u_truth)) / 100) 
        cs = ax.contourf(X1.reshape(dim,dim), X2.reshape(dim,dim), u_truth.reshape(dim,dim), levels,cmap=plt.get_cmap('Spectral'))
        cbar = fig.colorbar(cs)
        plt.xticks([-1,-0.5,0,0.5,1])
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.title('$u$(truth)')
        if align:
            plt.savefig(output_path + "u_truth_aligned.png", format="png", dpi=100, bbox_inches="tight")
        else:
            plt.savefig(output_path + "u_truth.png", format="png", dpi=100, bbox_inches="tight")