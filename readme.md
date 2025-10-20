# Deep Neural ODE Operator Networks for PDEs

**NODE-ONet** (deep neural ordinary differential equation operator network) is a framework for learning solution operators of partial differential equations (PDEs). By integrating neural ODEs (NODEs) within an encoder-decoder architecture, the NODE-ONet framework effectively decouples spatial and temporal variables, aligning with traditional numerical methods for time-dependent PDEs.  This repository provides a PyTorch implementation of NODE-ONet and compares its performance against DeepONet and MIONet on various simulation tasks. 
The source code is for the paper: [Z. Li, K. Liu, Y. Song, H. Yue, E. Zuazua. Deep Neural ODE Operator Networks for PDEs, arXiv: 2510.15651, 2025](https://arxiv.org/abs/2510.15651)

## Installation

To get started, clone this repository and install the required dependencies. We recommend using a virtual environment. The required packages are written in `requirements.txt` file.

## Usage

This repository includes several demo scripts to run experiments. Each demo can be executed directly and does not require additional configuration. The scripts will generate simulation data, train the models, and produce output metrics/plots. Use the following commands to run each experiment:

**(1) A 1D nonlinear diffusion-reaction equation**
```math
\begin{cases}
\partial_t u(t,x)-\nabla\!\cdot\!\big(D(t,x)\nabla u(t,x)\big)+R(t,x)\,u^2(t,x)=f(t,x), & (t,x)\in[0,T]\times\Omega,\\
u(0,x)=u_0(x), & x\in\Omega,\\
u(t,x)=u_b(t,x), & (t,x)\in[0,T]\times\partial\Omega.
\end{cases}
```

- **Learn the source-to-solution operator $\Psi_f^{\dagger}:f\mapsto u$:**
  Run `1D_source.py` to train the NODE-ONet model to learn the source-to-solution operator $\Psi_f^\dagger: f\mapsto u$. 

- **Learn the diffusion-to-solution operator $\Psi_D^\dagger: D\mapsto u$:**
  Run `1D_diffusion.py` to train the NODE-ONet model to learn the diffusion-to-solution operator $\Psi_D^\dagger: D\mapsto u$. 

- **Learn the multi-input operator $\Psi_m^\dagger: \{D,f\}\mapsto u$:**
  Run `1D_multi_input.py` to train the NODE-ONet model to Learn the multi-input operator $\Psi_m^\dagger: \{D,f\}\mapsto u$.

- **Generalization capacity of ${\alpha}$:**
  Run `1D_generalization.py` to train the NODE-ONet to learn the source-to-solution operator $\Psi_f^\dagger$ with $D=0.01, R=-0.01$. Then learn the source-to-solution operator with $D=0.2$, $R=0$ by the NODE-ONet with the pre-trained neural network $N_{\theta_{\alpha}^*}$.  

- **Prediction beyond the training time frame:**

  Run `1D_predict.py` to train the NODE-ONet to learn the source-to-solution operator $\Psi_f^*$ and the multi-input operator $\Psi_m^*$ with training time $[0,1]$ and testing time $[0,2]$.

- **Flexibility of encoder/decoder:**

  Run `1D_source_Fourier.py` to train the NODE-ONet to learn the source-to-solution operator $\Psi_f: f\mapsto u$ by Fourier basis functions.

**(2) A 2D Navier-Stokes equation**
```math
\begin{cases}
\partial_t u(t,x)+\mathbf{V}(t,x)\cdot\nabla u(t,x)=\nu\,\Delta u(t,x)+f(t,x), & (t,x)\in[0,T]\times\Omega,\\
u(t,x)=\nabla\times\mathbf{V}(t,x)\coloneqq \partial_{x_1} V_2-\partial_{x_2} V_1, & (t,x)\in[0,T]\times\Omega,\\
\nabla\cdot\mathbf{V}(t,x)=0, & (t,x)\in[0,T]\times\Omega,\\
u(x,0)=u_0(x), & x\in\Omega .
\end{cases}
```


Run `2D_NS.py` to train the NODE-ONet to learn the following three operators:

- **The initial value-to-solution operator $\Psi_i: u_0\mapsto u$** with the fixed source term $f(x_1,x_2)=0.1 \sin(2\pi(x_1 + x_2)) + 0.1 \cos(2\pi(x_1 + x_2))$;
- **The source-to-solution operator $\Psi_f: f\mapsto u$** with the fixed initial value $u_0(x_1,x_2)=0.1 \sin(2\pi(x_1 + x_2)) + 0.1 \cos(2\pi(x_1 + x_2))$;
- **The solution operator with multi-input $\Psi_m: \{u_0,f\}\mapsto u$**.

## Citation

If you use this project in your research, please cite us using the following BibTeX entry:

```bibtex
@misc{li2025deepneuralodeoperator,
      title={Deep Neural ODE Operator Networks for PDEs}, 
      author={Ziqian Li and Kang Liu and Yongcun Song and Hangrui Yue and Enrique Zuazua},
      year={2025},
      eprint={2510.15651},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.15651}, 
}
```