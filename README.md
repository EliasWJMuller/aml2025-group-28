# FS25_AML_Group_28 Semester Project
Advanced Machine Learning Course FS25 @ UZH; Group 28; RNA-3D folding with Diffusion Denoising Implicit Models (DDIM) by Diyar Taskiran, Niklas Schmidt and Elias Müller.

## Short Overview
This project investigates the potential of integrating Diffusion Denoising Implicit Models (DDIM) with the RNAgrail framework for predicting the 3D structures of RNA. The goal is to improve the efficiency at inference while preserving the quality of the predicted structures.

## Installation
In order to run the code it its original form, RNAgrail requires a CUDA-enabled GPU with sufficient memory. It is still possible to run the code either on CPU or on another nn backend, such as MPS, yet requires significant changes to the setup, outlined in "Installation_Instructions.md". The current state of the code is compatible with MPS.