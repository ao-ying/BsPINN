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

# 系统设置
torch.set_default_dtype(torch.float)
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Device configuration
device_cpu = torch.device('cpu')
path = os.path.dirname(__file__) + "/"
torch.backends.cudnn.benchmark = True
plt.rcParams["text.usetex"] = True
plt.rcParams['font.size'] = 30

# 精确解
# x待求值点的d维向量，若有n个点待求则x的维度为n*d
def solution(x):
    if len(x.shape) > 1: # x为n*d维向量
        x_sum = np.sum(x, axis=1)
        x_mean = np.mean(x, axis=1)
    else :
        x_sum = np.sum(x) # x为d维向量
        x_mean = np.mean(x)
    ustar = x_mean ** 2 + np.sin(coe * x_sum) # x_mean ** 2
    return ustar

# 求偏导
def grad(f,x):
    ret = torch.autograd.grad(f, x, torch.ones_like(f).to(device), retain_graph=True, create_graph=True)[0]
    return ret

# 输出模型参数信息
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.params)
    trainable_num = sum(p.numel() for p in net.params if p.requires_grad)
    # print("Total: %d, Trainable: %d" % (total_num, trainable_num))
    return trainable_num
    
#  将优化器optimizer的学习率设置为lr
def reset_lr(optimizer, lr):
    for params in optimizer.param_groups: 
            params['lr'] = lr

# 生成数据集
# lb_x：x中每个元素的下界，ub_x：x中每个元素的上界，dim：x的维度，num_sample求解域内采样点个数，num_boundary边界上采样点个数(每个维度的边界采num_boundary/dim个点，上下边界各一半)
def load_data(lb_geom, ub_geom, num_sample, num_boundary, dim_x):
    # 训练集
    ## 区域内
    X_train = np.random.uniform(lb_geom, ub_geom, (num_sample, dim_x))
    ## 边界
    num_per_dim = int(num_boundary / dim_x)
    tool = np.zeros(num_per_dim)
    tool[:int(num_per_dim/2)] = ub_geom
    tool[int(num_per_dim/2):] = lb_geom
    for i in range(dim_x):
        boundary_points = np.random.uniform(lb_geom, ub_geom, (num_per_dim, dim_x))
        boundary_points[:, i] = tool
        X_train = np.r_[X_train, boundary_points]
    
    return X_train

# 残差块
class Residual(nn.Module):
    def __init__(self, Layers):
        super().__init__()
        # 数据初始化
        self.Layers = copy.deepcopy(Layers)
        self.Layers.insert(0, Layers[0]) # 与普通二分神经网络比，少了最后一层。
        self.num_Layers = len(self.Layers)
        self.width = [self.Layers[0]] + [int(pow(2, i - 1) * self.Layers[i]) for i in range(1, len(self.Layers))] # 与普通二分神经网络比，少了最后一层。
        self.masks = self.construct_mask()
        self.num_params = self.cal_param()
        
        # 权重初始化
        self.params = []
        # 初始化二分神经网络
        self.weights, self.biases =  self.initialize_NN(self.Layers)
        
    # 计算参数个数
    def cal_param(self):
        ret = 0
        for i in range(self.num_Layers - 1):
            temp = pow(2, i) * (self.Layers[i] * self.Layers[i + 1] + self.Layers[i + 1])
            ret += temp
        return ret
    
    # 创建掩码
    def construct_mask(self):
        masks = []
        for l in range(2, self.num_Layers - 1):
            # 计算块矩阵维度
            num_blocks = int(pow(2, l - 1))
            blocksize1 = int(self.width[l] / num_blocks)
            blocksize2 = 2 * self.Layers[l + 1]
            blocks = [torch.ones((blocksize1,blocksize2)) for i in range(num_blocks)]
            mask = torch.block_diag(*blocks).to(device)
            masks.append(mask)
        return masks
    
    # 初始化二分神经网络参数
    def initialize_NN(self, Layers):                     
        weights = []
        biases = []
        # 第一个隐藏层
        tempw = torch.zeros(self.Layers[0], self.width[1]).to(device)
        w = Parameter(tempw, requires_grad=True)
        nn.init.xavier_uniform_(w, gain=1) 
        tempb = torch.zeros(1, self.width[1]).to(device)
        b = Parameter(tempb, requires_grad=True)
        weights.append(w)
        biases.append(b) 
        self.params.append(w)
        self.params.append(b)
        # 中间的隐藏层
        for l in range(1,self.num_Layers - 1):
            # 权重w
            tempw = torch.zeros(self.width[l], self.width[l+1]).to(device)
            # 块对角矩阵初始化
            for i in range(int(pow(2,l))): # 遍历每个小矩阵并初始化
                tempw2 = torch.zeros(Layers[l], Layers[l+1])
                w2 = Parameter(tempw2, requires_grad=False)
                nn.init.xavier_uniform_(w2, gain=1)  
                row_index = int(i / 2)
                tempw[row_index * Layers[l] : (row_index + 1) * Layers[l], i * Layers[l+1] : (i + 1) * Layers[l+1]] = w2.data
            w = Parameter(tempw, requires_grad=True)
            # 偏置b
            tempb = torch.zeros(1, self.width[l+1]).to(device)
            b = Parameter(tempb, requires_grad=True)
            weights.append(w)
            biases.append(b) 
            self.params.append(w)
            self.params.append(b)
        return weights, biases
        
    def forward(self, X):
        H = X
        # 神经网络部分
        for l in range(0, self.num_Layers - 1):
            if l >=2 and l <= self.num_Layers - 2:
                W = self.weights[l]
                W2 = W * self.masks[l - 2]
                b = self.biases[l]
                H = torch.add(torch.matmul(H, W2), b)
                # 若不是最后一个全连接层则有激活函数(之前没有这个判断)
                if l != self.num_Layers - 2:
                    H = torch.sin(H)
            else:
                W = self.weights[l]
                b = self.biases[l]
                H = torch.add(torch.matmul(H, W), b)
                H = torch.sin(H) 
        # 残差和  
        Y = torch.sin(torch.add(H, X))
        return Y
    
    def forward2(self, X):
        H = X
        # 神经网络部分
        for l in range(0, self.num_Layers - 1):
            if l >=2 and l <= self.num_Layers - 2:
                W = self.weights2[l]
                W2 = W * self.masks2[l - 2]
                b = self.biases2[l]
                H = torch.add(torch.matmul(H, W2), b)
                # 若不是最后一个全连接层则有激活函数(之前没有这个判断)
                if l != self.num_Layers - 2:
                    H = torch.sin(H)
            else:
                W = self.weights2[l]
                b = self.biases2[l]
                H = torch.add(torch.matmul(H, W), b)
                H = torch.sin(H) 
        # 残差和  
        Y = torch.sin(torch.add(H, X))
        return Y
    
    def set_device(self, device):
        # weight, biases
        self.weights2 = [0 for i in range(self.num_Layers - 1)]
        self.biases2 = [0 for i in range(self.num_Layers - 1)]
        for l in range(0, self.num_Layers - 1):
            self.weights2[l] = self.weights[l].data
            self.weights2[l] = self.weights2[l].to(device)
            self.biases2[l] = self.biases[l].data
            self.biases2[l] = self.biases2[l].to(device)
        # mask
        self.masks2 = [0 for i in range(self.num_Layers - 3)]
        for l in range(0, self.num_Layers - 3):
            self.masks2[l] = self.masks[l].data
            self.masks2[l] = self.masks2[l].to(device)    
    
    # def set_device(self, device):
    #     for l in range(0, self.num_Layers - 1):
    #         self.weights[l] = self.weights[l].to(device)
    #         self.biases[l] = self.biases[l].to(device)
    #     for l in range(0, self.num_Layers - 3):
    #         self.masks[l] = self.masks[l].to(device)

# PINN神经网络
class PINN(nn.Module):
    def __init__(self, Layers, num_sample, num_boundary, boundary_weight, lb_X, ub_X, num_res, dim_x, dim_y, coe):
        super(PINN, self).__init__()
        # 初始化参数
        self.lb_X = lb_X
        self.ub_X = ub_X
        self.Layers = Layers
        self.num_sample = num_sample
        self.boundary_weight = boundary_weight
        self.num_res = num_res
        self.params = []
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_res_fc = Layers[0] # 残差块中全连接网络宽度，每层宽度都相等
        self.num_boundary = num_boundary
        self.coe = coe
        # 初始化第一个全连接层
        self.fc_first = nn.Linear(dim_x, self.dim_res_fc)
        nn.init.xavier_uniform_(self.fc_first.weight, gain=1)
        nn.init.zeros_(self.fc_first.bias)
        self.params.append(self.fc_first.weight)
        self.params.append(self.fc_first.bias)
        # 初始化残差层
        self.res_blocks = nn.ModuleList([Residual(self.Layers) for i in range(self.num_res)])
        for i in range(self.num_res):
            self.params.extend(self.res_blocks[i].params)
        # 初始化最后一个全连接层
        self.fc_last = nn.Linear(self.dim_res_fc, self.dim_y)
        nn.init.xavier_uniform_(self.fc_last.weight, gain=1)
        nn.init.zeros_(self.fc_last.bias)
        self.params.append(self.fc_last.weight)
        self.params.append(self.fc_last.bias)
        # 计算参数
        self.num_params = sum(p.numel() for p in self.fc_first.parameters() if p.requires_grad)
        for n in range(self.num_res):
            self.num_params += self.res_blocks[i].num_params
        self.num_params += sum(p.numel() for p in self.fc_last.parameters() if p.requires_grad)
    
    # 全连接神经网络部分
    def neural_net(self, X):
        # 数据预处理，这里训练和测试时用的lb和ub应该是一样的，否则训练和测试用的神经网络就不一样了。
        X = 2.0 * (X - self.lb_X) / (self.ub_X - self.lb_X) - 1.0
        H = X.float()
        # ResNet部分
        H = self.fc_first(H)
        for i in range(self.num_res):
            H = self.res_blocks[i](H)
        Y = self.fc_last(H)

        return Y

    def neural_net2(self, X):
        # 数据预处理，这里训练和测试时用的lb和ub应该是一样的，否则训练和测试用的神经网络就不一样了。
        X = 2.0 * (X - self.lb_X2) / (self.ub_X2 - self.lb_X2) - 1.0
        H = X.float()
        # 神经网络部分
        H= torch.add(torch.matmul(H, self.wf2), self.bf2)
        for i in range(self.num_res): 
            H = self.res_blocks[i].forward2(H)
        Y = torch.add(torch.matmul(H, self.wl2), self.bl2)

        return Y

    # PDE部分
    def he_net(self, X):
        # 方程
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
        f = self.dim_x * (self.coe ** 2) * torch.sin(self.coe * sum_x_e) - 2 / self.dim_x # 方程右端项
        equation = - Laplace_u - f

        # 边界条件
        X_b = [0 for i in range(self.dim_x)]
        for i in range(self.dim_x):
            X_b[i] = X[self.num_sample:, i : i + 1].clone()
        u_b = self.neural_net(torch.cat(X_b, dim = 1)) # self.num_boundary * 1
        sum_x_b = torch.sum(X[self.num_sample:, :], dim=1, keepdim=True) # self.num_boundary * 1
        mean_x_b = sum_x_b / self.dim_x
        Dvalue = torch.pow(mean_x_b, 2) + torch.sin(self.coe * sum_x_b) # 狄利克雷边值
        boundary = u_b - Dvalue

        # 总位移
        u = torch.cat([u_e, u_b], dim = 0) # (self.num_sample + self.num_boundary) * 1

        return u, equation, boundary

    # 损失函数
    def loss(self,X_train):
        # 计算方程和边界条件项
        _, equation, boundary = self.he_net(X_train)
        
        # 计算总误差
        loss_e = torch.mean(torch.square(equation))
        loss_b = torch.mean(torch.square(boundary))
        loss_all = loss_e + self.boundary_weight * loss_b

        return loss_all
    
    # 预测X对应点处的u值
    def predict(self, X):
        u_pred = self.neural_net(X)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred 
    
    def predict2(self, X):
        u_pred = self.neural_net2(X)
        u_pred = u_pred.cpu().detach().numpy()
        return u_pred 
    
    # 应该将参数用.data赋值给一个新的变量，之后再用.to(device)
    def set_device(self, device): 
        # 归一化变量
        self.lb_X2 = self.lb_X.data
        self.lb_X2 = self.lb_X2.to(device)
        self.ub_X2 = self.ub_X.data
        self.ub_X2 = self.ub_X2.to(device)
        # 第一个全连接层
        self.wf2 = self.fc_first.weight.data.T
        self.wf2 = self.wf2.to(device)
        self.bf2 = self.fc_first.bias.data
        self.bf2 = self.bf2.to(device)
        # 残差块
        for i in range(self.num_res):
            self.res_blocks[i].set_device(device)
        # 最后一个全连接层
        self.wl2 = self.fc_last.weight.data.T
        self.wl2 = self.wl2.to(device)
        self.bl2 = self.fc_last.bias.data
        self.bl2 = self.bl2.to(device)
        
    # 计算相对误差: 每次计算大约需要5s
    def rel_error(self):
        # 数据转换1
        self.set_device(device_cpu)
        
        # 计算误差
        u_pred = self.predict2(X_pred).flatten()
        u_truth = solution(points)
        u_L2RE = np.sum(weights * np.square(u_pred - u_truth)) / np.sum((weights * np.square(u_truth))) 

        return u_L2RE

# main函数
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
        # 设置随机种子
        np.random.seed(seed)
        torch.manual_seed(seed)
        # 参数设置
        ## 方程相关
        lb_geom = -1 # x下界
        ub_geom = 1 # x上界
        dim_x = 10 # x维度
        dim_y = 1 # y维度
        coe = 0.6 * np.pi # sin中的系数
        
        ## 神经网络相关
        name = "BsPINN_256-32_Adam_v1"
        num_sample = 4000
        num_boundary = 2000 # 必须整除dim_x
        epochs = 10000 # 10000 # 优化器迭代次数
        Layers = [256, 128, 64, 32] # 一个残差块中全连接网络结构，各个层神经元个数必须相等
        learning_rate = 0.001 # 初始学习率 
        boundary_weight = 1 # 设置太大可能导致梯度爆炸
        num_res = 2 # 残差块数量
        h = int(1000) # 学习率衰减相关
        r = 0.95 # 学习率每隔h轮迭代乘r
        weight_decay = 0.001 # L2正则项系数，防止损失值突增
        
        ## 画图相关
        train = False
        align = False
        lb_loss = 1e-3
        ub_loss = 1e21
        lb_error = 1e-2
        ub_error = 1e1
        lb_u = -1.1
        ub_u = 1.1
        lb_diff = -0.05
        ub_diff = 0.05
        error_param = 5 # 画出(20 * error_param)个误差点，每画一个大约消耗6s。epochs需要可以被(20 * error_param)整除。
        record_error = False
        
        # 辅助变量
        name = name + ("_%d" % epochs)
        print("\n\n***** name = %s *****" % name)
        print("seed = %d" % seed)
        output_path = path + ('./output_BsPINN/')
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('./output_BsPINN/%s/' % name)
        if not os.path.exists(output_path): os.mkdir(output_path)
        output_path = path + ('./output_BsPINN/%s/train_%d/' % (name, seed))
        if not os.path.exists(output_path): os.mkdir(output_path)
        
        # 高斯积分数据(不要设为self.变量，否则将导致torch.save很慢)
        # dim_gauss = 4
        # points = np.loadtxt(path + "../data/gauss_points_%d.txt" % dim_gauss, dtype = float, delimiter=' ')
        # weights = np.loadtxt(path + "../data/gauss_weights_%d.txt" % dim_gauss, dtype = float, delimiter=' ')
        X_pred = torch.from_numpy(points).float().to(device_cpu)

        # 生成数据集
        if train:
            print("Loading data")
            ## 训练数据
            X_train = load_data(lb_geom, ub_geom, num_sample, num_boundary, dim_x)
            lb_X = X_train.min(0) # 得到训练集每个维度上的最大值组成的元素
            ub_X = X_train.max(0) # 得到训练集每个维度上的最小值组成的元素
            lb_X = torch.from_numpy(lb_X).float().to(device)
            ub_X = torch.from_numpy(ub_X).float().to(device)
            X_train = torch.from_numpy(X_train).float().to(device)
            
            # 声明神经网络实例
            model = PINN(Layers, num_sample, num_boundary, boundary_weight, lb_X, ub_X, num_res, dim_x, dim_y, coe)
            model = nn.DataParallel(model)
            model = model.module
            model.to(device)
            print(model) # 打印网络概要
            params = model.num_params
            print("params = %d" % params)
            
            # 优化器
            optimizer = optim.Adam(model.params, lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=weight_decay, amsgrad=False)

            # 训练
            start = time.time()
            ## adam
            loss_list = []
            error_list = [] # 保存在测试集上的平均相对误差值
            min_loss = 9999999
            print("Start training!")
            for it in range(epochs):
                # 优化器训练
                Loss = model.loss(X_train)
                optimizer.zero_grad(set_to_none=True)
                Loss.backward() 
                optimizer.step()

                # 衰减学习率
                if (it + 1) % h == 0: # 指数衰减
                    learning_rate *= r
                    reset_lr(optimizer, learning_rate)

                # 保存损失值和相对误差，并保存训练损失最小的模型
                loss_val = Loss.cpu().detach().numpy() 
                loss_list.append(loss_val)
                
                # save the model with the minimum train loss
                if loss_val < min_loss: # 这一步导致BsPINN比PINN慢
                    torch.save(model, output_path + 'network.pkl') 
                    min_loss = loss_val
                        
                # 保存误差曲线
                if record_error and (it + 1) % (epochs/20/error_param) == 0:
                    u_L2RE = model.rel_error() # 每次消耗6s
                    error_list.append(u_L2RE)

                # 输出
                if (it + 1) % (epochs/20) == 0:
                    if record_error:
                        print("It = %d, loss = %.8f, u_L2RE = %.8f, finish: %d%%" % ((it + 1), loss_val, u_L2RE, (it + 1) / epochs * 100))
                    else:
                        print("It = %d, loss = %.8f, finish: %d%%" % ((it + 1), loss_val, (it + 1) / epochs * 100))
                
            ## 后续处理
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

        # 保存loss曲线
        plt.rcParams['font.size'] = 20
        loss_list = np.loadtxt(output_path + "loss.txt", dtype = float, delimiter=' ')
        plt.figure()
        plt.semilogy(loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if align:
            plt.ylim(lb_loss, ub_loss) # 与BiPINN统一量纲
            plt.savefig(output_path + 'loss_aligned.pdf', format="pdf", dpi=100, bbox_inches="tight")
        else:
            plt.savefig(output_path + 'loss.pdf', format="pdf", dpi=100, bbox_inches="tight")
            
        # 保存error曲线
        if record_error:
            fig, ax = plt.subplots()
            error_list = np.loadtxt(output_path + "error.txt", dtype = float, delimiter=' ')
            plt.semilogy(error_list)
            plt.xlabel('Epoch')
            plt.ylabel('Relative error')
            tool = [0, 4, 8, 12, 16, 20]
            plt.xticks([tool[i] * error_param for i in range(len(tool))], [str(0), str(int(epochs * 0.2)), str(int(epochs * 0.4)), str(int(epochs * 0.6)), str(int(epochs * 0.8)), str(int(epochs))])
            if align:
                plt.ylim(lb_error, ub_error) # 与BiPINN统一量纲
                plt.savefig(output_path + 'error_aligned.pdf', format="pdf", dpi=300, bbox_inches="tight")
            else:
                plt.savefig(output_path + 'error.pdf', format="pdf", dpi=300, bbox_inches="tight")

        # 计算网格点上u的相对误差
        # dim_gauss = 4
        X_pred = torch.from_numpy(points).float().to(device_cpu)
        model2 = torch.load(output_path + 'network.pkl', map_location=device_cpu) 
        u_pred = model2.predict(X_pred).flatten()
        u_truth = solution(points)
        u_L2RE = np.sum(weights * np.square(u_pred - u_truth)) / np.sum((weights * np.square(u_truth)))
        print("u_L2RE = %.10f when using Gauss L2RE" % u_L2RE)
        np.savetxt(output_path + "u_truth.txt", u_truth, fmt="%s", delimiter=' ')
        np.savetxt(output_path + "u_pred_bipinn.txt", u_pred, fmt="%s", delimiter=' ')
        
        # 画图数据准备
        dim = 500
        X1 = np.linspace(lb_geom, ub_geom, dim)
        X2 = np.linspace(lb_geom, ub_geom, dim)
        X1, X2 = np.meshgrid(X1, X2)
        X1 = X1.flatten().reshape(dim*dim,1)
        X2 = X2.flatten().reshape(dim*dim,1)
        points2 = np.c_[X1, X2] # N * 2
        tool = np.zeros((X1.shape[0], dim_x - 2)) # 后8个元素应该全是1
        points2 = np.c_[points2, tool] # N * dim_x，N个测试点。
        u_truth = solution(points2)
        X_pred2 = points2
        X_pred2 = torch.from_numpy(X_pred2).float().to(device_cpu)
        model2 = torch.load(output_path + 'network.pkl', map_location=device_cpu) 
        if train: model2.set_device(device_cpu)
        u_pred = model2.predict(X_pred2).flatten()
        u_L2RE2 = np.linalg.norm((u_truth - u_pred),ord=2) / (np.linalg.norm(u_truth, ord=2))
        # print("u_L2RE = %.8f when drawing." % u_L2RE2)

        # 画预测解图像
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
            
        # 画精确解图像
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
            
        # 画误差图像
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
            
        # 保存误差值
        csv_data.append([seed, u_L2RE])
    
    # 保存为csv文件
    output_path = path + ('./output_BsPINN/%s/' % name)
    file_write = open(output_path + 'record.csv','w')
    writer = csv.writer(file_write)
    writer.writerow(['seed','L2RE'])
    writer.writerows(csv_data)
    file_write.close()