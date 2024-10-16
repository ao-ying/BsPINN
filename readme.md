# Binary structured physics-informed neural networks for solving equations with rapidly changing solutions
Physics-informed neural networks (PINNs), rooted in deep learning, have emerged as a promising approach for solving partial differential equations (PDEs). By embedding the physical information described by PDEs into feedforward neural networks, PINNs are trained as surrogate models to approximate solutions without the need for label data. Nevertheless, even though PINNs have shown remarkable performance, they can face difficulties, especially when dealing with equations featuring rapidly changing solutions. These difficulties encompass slow convergence, susceptibility to becoming trapped in local minima, and reduced solution accuracy. To address these issues, we propose a binary structured physics-informed neural network (BsPINN) framework, which employs binary structured neural network (BsNN) as the neural network component. By leveraging a binary structure that reduces inter-neuron connections compared to fully connected neural networks, BsPINNs excel in capturing the local features of solutions more effectively and efficiently. These features are particularly crucial for learning the rapidly changing in the nature of solutions. In a series of numerical experiments solving the Euler equation, Burgers equation, Helmholtz equation, and high-dimension Poisson equation, BsPINNs exhibit superior convergence speed and heightened accuracy compared to PINNs. \textcolor[rgb]{0.00,0.7,0.00}{Additionally, we compare BsPINNs with XPINNs, FBPINNs and FourierPINNs, finding that BsPINNs achieve the highest or comparable results in many experiments. Furthermore, our experiments reveal that integrating BsPINNs with Fourier feature mappings results in the most accurate predicted solutions when solving the Burgers equation and the Helmholtz equation on a 3D M\"{o}bius knot. This demonstrates the potential of BsPINNs as a foundational model.} From these experiments, we discover that BsPINNs resolve the issues caused by increased hidden layers in PINNs resulting in over-fitting, and prevent the decline in accuracy due to non-smoothness of PDEs solutions.

For more information, please refer to the following: 
+ Liu Y, Wu R, Jiang Y. Binary structured physics-informed neural networks for solving equations with rapidly changing solutions, J. Comput. Phys. 518(2024)113341.(https://www.sciencedirect.com/science/article/pii/S0021999124005898)

# Environment
```txt
python 3.9
torch 2.0.1
numpy 1.24.1
CUDA 12.2 
matplotlib 3.4.3
modulus 22.09
```

# How to run the program
Run the program in the "python" folder in each equation folder, where `*_BsPINN.py` means using BsPINN to solve the equation, `*_FourierBsPINN.py` means using FourierBsPINN to solve the equation. 

# Results
Network models, images and other files obtained from the program will be saved in a `output_*` folder in the directory where the python program is located.

# Citation
```bibtex
@article{liu2024binary,
  title={Binary structured physics-informed neural networks for solving equations with rapidly changing solutions},
  author={Liu, Yanzhi and Wu, Ruifan and Jiang, Ying},
  journal={J. Comput. Phys.},
  volume={518},
  pages={113341},
  publisher={Elsevier}
}
```