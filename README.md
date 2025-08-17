# Mamba Neural Operator for Linear PDEs

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Abstract

**Mamba** is a high-performance architecture based on state space models, specifically designed to run efficiently on modern GPUs, unlike other neural network components that can be executed on standard CPUs.  
The Mamba block excels at modeling long-range sequences in linear time, and this capability is leveraged for the numerical resolution of **linear partial differential equations (PDEs)**.  

Three case studies respectively addressed elliptic, parabolic, and hyperbolic PDEs. Simulation results demonstrate that Mamba-based neural networks can successfully learn the underlying differential operators.
However, limitations were observed due to restricted datasets, relatively simple architectures, and limited computational resources. 
Despite these, the experiments provide a solid foundation for future improvements, including nonlinear PDEs (e.g., Navier–Stokes equations) and real-time simulation of high-frequency time series.

---

## Installation

Make sure you have a working CUDA/PyTorch environment. Then follow the steps below:

```bash
# 1) Clean any old/broken installs
pip uninstall -y mamba-ssm selective_scan

# 2) Build essentials for CUDA/C++ extensions
pip install --upgrade pip setuptools wheel ninja cmake packaging

# 3) (optional but recommended) compile settings
export MAX_JOBS=$(nproc)
# Choose arches you actually use; these cover A100(8.0), 30xx(8.6), 4090(8.9), H100(9.0)
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"

# 4) Install from source (forces local compile against your PyTorch/CUDA)
pip install --no-cache-dir --no-build-isolation git+https://github.com/state-spaces/mamba.git

# 5) Install additional dependencies
pip install matplotlib
