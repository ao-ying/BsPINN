# Binary structured physics-informed neural networks for solving equations with rapidly changing solutions
Physics-informed neural networks (PINNs), rooted in deep learning, have emerged as a promising approach for solving partial differential equations (PDEs). By embedding the physical information described by PDEs into feedforward neural networks, PINNs are trained as surrogate models to approximate solutions without the need for label data. Nevertheless, even though PINNs have shown remarkable performance, they can face difficulties, especially when dealing with equations featuring rapidly changing solutions. These difficulties encompass slow convergence, susceptibility to becoming trapped in local minima, and reduced solution accuracy. To address these issues, we propose a binary structured physics-informed neural network (BsPINN) framework, which employs binary structured neural network (BsNN) as the neural network component. By leveraging a binary structure that reduces inter-neuron connections compared to fully connected neural networks, BsPINNs excel in capturing the local features of solutions more effectively and efficiently. These features are particularly crucial for learning the rapidly changing in the nature of solutions. In a series of numerical experiments solving Burgers equation, Euler equation, Helmholtz equation, and high-dimension Poisson equation, BsPINNs exhibit superior convergence speed and heightened accuracy compared to PINNs. From these experiments, we discover that BsPINNs resolve the issues caused by increased hidden layers in PINNs resulting in over-smoothing, and prevent the decline in accuracy due to non-smoothness of PDEs solutions.

# Environment
```txt
python 3.9.13
torch 2.0.1
numpy 1.24.1
CUDA 12.2 
matplotlib 3.4.3
```

# Citation
```bibtex
@article{liu2024binary,
  title={Binary structured physics-informed neural networks for solving equations with rapidly changing solutions},
  author={Liu, Yanzhi and Wu, Ruifan and Jiang, Ying},
  journal={arXiv preprint arXiv:2401.12806},
  year={2024}
}
```