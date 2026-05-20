# mnist47266111
UTokyo mnist homework.

This repository contains a minimal Denoising Diffusion Probabilistic Model
(DDPM) experiment on the MNIST handwritten digit dataset.

The main script, `mnist_ddpm.py`, trains a small time-conditioned CNN/U-Net
denoiser to predict the Gaussian noise added to MNIST images at different
diffusion timesteps. After training, the script samples new digit images from
random noise and saves several visualization results.

## Files

- `mnist_ddpm.py`  
  Main training and sampling script. It includes the forward noising process,
  the reverse denoising process, a small time-conditioned U-Net, training code,
  and image/checkpoint saving logic.

- `checkpoint.pt`  
  Saved model checkpoint after training.

- `forward_noising_grid.png`  
  Visualization of the forward diffusion process. A clean MNIST digit is
  gradually corrupted by Gaussian noise.

- `reverse_denoising_trajectory.png`  
  Visualization of the reverse sampling process. The model starts from random
  noise and gradually denoises it into a digit-like image.

- `generated_digits_grid.png`  
  A grid of generated MNIST-style digit samples produced by the trained DDPM.

- `loss_curve.png`  
  Training loss curve for the noise-prediction objective.

## Experiment Setup

- Dataset: MNIST, 28x28 grayscale images
- Normalization: images are scaled to `[-1, 1]`
- Diffusion steps: `T = 200`
- Noise schedule: linear beta schedule
- Objective: predict the added Gaussian noise using MSE loss
- Model: small time-conditioned U-Net/CNN with sinusoidal timestep embeddings

## Result Summary

The model successfully learns the basic DDPM denoising objective. The loss curve
decreases clearly during training, the forward grid shows the expected noising
process, and the reverse trajectory shows digit-like structure gradually
emerging from noise. The generated samples are recognizable as MNIST-style
digits, although some remain slightly noisy or ambiguous because this is a small
CPU-friendly experiment.
