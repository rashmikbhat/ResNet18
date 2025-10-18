# Hyperparameter Analysis of ResNet-18 on CIFAR-10

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive empirical investigation of hyperparameter effects on ResNet-18 architecture for CIFAR-10 image classification. This project systematically analyzes learning rates, scheduling strategies, weight decay regularization, and custom batch normalization implementations.

## 📋 Project Overview

This repository contains the complete implementation and report for analyzing hyperparameter effects on ResNet-18 trained on CIFAR-10 dataset. The project follows a rigorous experimental methodology with proper train-validation-test splits and provides insights into optimization-regularization trade-offs in deep learning.

### 🎯 Key Objectives

- Analyze the impact of initial learning rates (0.1, 0.01, 0.001)
- Compare learning rate scheduling strategies (Constant vs Cosine Annealing)
- Investigate weight decay regularization effects (5e-4, 1e-2)
- Implement and evaluate custom batch normalization
- Provide practical guidelines for deep neural network training
