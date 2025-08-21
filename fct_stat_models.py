import torch
import numpy as np
from sklearn.decomposition import FactorAnalysis

import fct_facilities as fac 


tol = 1e-1
steps_print = 2
decayRate_general = 1 # 0.96

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

#### Code based on:

"""
Created on Thu Sep 8 16:45:52 2022

Statistical models for fitting single-trial neuronal population responses
from one brain area to multiple stimuli. These models are designed to
capture trial-to-trial variability in the neuronal population response.

@author: xiaji
"""

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

# Set the device for PyTorch operations.
# 'cpu' is used for CPU computation. 'cuda' can be used for GPU if available.
dev = "cpu"
device = torch.device(dev)


####################################
class multiplicative_model():
    """
    Implements a multiplicative model for neuronal population responses.
    This model assumes that the shared variability (latent factors) scales
    multiplicatively with the stimulus-specific mean response.

    Model: x_st = d_s + A_s * z_st + epsilon_st
    where A_s = alpha_p * diag(d_s), with d_s being the stimulus-specific mean,
    alpha_p being a global loading matrix, z_st are latent variables, and
    epsilon_st is private noise.
    """

    def __init__(self,x, n, n_stim, n_trial, n_compo, alpha_p_init, psi_p_init):
        """
        Initializes the multiplicative_model.

        Args:
            x (np.ndarray): Input data (neuronal responses) with shape
                            (n_neurons, n_stimuli * n_trials).
            n (int): Number of neurons.
            n_stim (int): Number of stimuli.
            n_trial (int): Number of trials per stimulus.
            n_compo (int): Number of latent components (dimensions of z).
            alpha_p_init (np.ndarray): Initial guess for the global loading
                                       matrix 'alpha_p'. Shape: (n_neurons, n_compo).
            psi_p_init (np.ndarray): Initial guess for the diagonal of the
                                      private noise covariance 'psi_p'. Shape: (n_neurons, n_stimuli).
        """
        self.n = n  # Number of neurons
        self.n_stim = n_stim  # Number of stimuli
        self.n_trial = n_trial  # Number of trials per stimulus
        self.x = x  # Input data
        self.n_compo = n_compo  # Number of latent components
        # A small constant to ensure positive definite covariance matrices and avoid numerical issues
        self.SMALL = 1e-5

        # Initialize parameters
        # d_p: Stimulus-specific mean responses, shape (n_neurons, n_stimuli)
        d_p = np.zeros((n, n_stim))
        # alpha_p: Global loading matrix, shape (n_neurons, n_compo)
        alpha_p = alpha_p_init
        # psi_p: Diagonal elements of private noise covariance, shape (n_neurons, n_stimuli)
        # Ensure psi_p is at least SMALL to avoid numerical issues (e.g., division by zero or log of zero)
        psi_p = np.maximum(psi_p_init, self.SMALL)

        # Calculate initial stimulus-specific mean responses (d_p) from the data
        for stim_i in range(n_stim):
            # Extract data for the current stimulus
            x_tmp = x[:, stim_i * n_trial:(stim_i + 1) * n_trial]
            # Calculate the mean response for each neuron for this stimulus
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)

        # Convert numpy arrays to PyTorch tensors and move them to the specified device
        self.d_p = torch.from_numpy(d_p).to(device)

        self.alpha_p = torch.from_numpy(alpha_p).to(device)
        self.alpha_p.requires_grad=True  # Enable gradient computation for alpha_p

        self.psi_p = torch.from_numpy(psi_p).to(device)
        self.psi_p.requires_grad=True  # Enable gradient computation for psi_p

    def loss_nll(self, x, n_trial):
        """
        Calculates the negative log-likelihood (NLL) of the model given data x.

        Args:
            x (np.ndarray): Data to calculate NLL for. Can be training or test data.
                            Shape: (n_neurons, n_stimuli * n_trials_data).
            n_trial (int): Number of trials per stimulus in x.

        Returns:
            torch.Tensor: The average negative log-likelihood across stimuli.
        """
        # Convert input data to PyTorch tensor and move to device
        x_var = torch.from_numpy(x).to(device)

        NLL = 0
        # Iterate through each stimulus to calculate NLL
        for stim_i in range(self.n_stim):
            # Extract stimulus-specific mean and private noise variance
            d_s = self.d_p[:, stim_i]
            # Ensure psi_s remains positive
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL, device=device))

            # Calculate the stimulus-specific loading matrix A
            # A = alpha_p * diag(d_s) is implemented as alpha_p * d_s[:, None]
            # d_s[:, None] reshapes d_s to (n_neurons, 1) for broadcasting
            A = self.alpha_p*d_s[:,None]
            # Calculate the full covariance matrix for the current stimulus
            # Covariance = A @ A.T + diag(psi_s)
            cov = A@A.T + torch.diag(psi_s)

            # Create a multivariate normal distribution object
            # The mean is d_s, and the covariance is 'cov'
            to_learn = torch.distributions.multivariate_normal.MultivariateNormal(loc=d_s, covariance_matrix= cov)
            # Calculate the negative mean log-probability for the current stimulus's data
            # x_var[:, stim_i*n_trial:(stim_i+1)*n_trial].T transposes the data
            # to be (n_trials_data, n_neurons) as expected by MultivariateNormal
            NLL += -torch.mean(to_learn.log_prob(x_var[:, stim_i*n_trial:(stim_i+1)*n_trial].T))

        # Return the average NLL across all stimuli
        return NLL/self.n_stim

    def recon_data(self, x, n_trial):
        """
        Reconstructs the data and estimates the posterior expectation of the latent variables (z).

        Args:
            x (np.ndarray): Data to reconstruct.
                            Shape: (n_neurons, n_stimuli * n_trials_data).
            n_trial (int): Number of trials per stimulus in x.

        Returns:
            tuple:
                - x_recon (torch.Tensor): Reconstructed data, shape (n_neurons, n_stimuli * n_trials_data).
                - E_z (torch.Tensor): Posterior expectation of latent variables E[z|x],
                                      shape (n_compo, n_stimuli * n_trials_data).
        """
        # Convert input data to PyTorch tensor and move to device
        x_var = torch.from_numpy(x).to(device)

        # Initialize tensors for estimated latent variables and reconstructed data
        E_z = torch.zeros([self.n_compo, self.n_stim*n_trial], dtype=torch.double).to(device)
        x_recon = torch.zeros([self.n, self.n_stim*n_trial]).to(device)

        for stim_i in range(self.n_stim):
            d_s = self.d_p[:, stim_i]
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL, device=device))
            x_s = x_var[:, stim_i*n_trial:(stim_i+1)*n_trial]

            A = self.alpha_p*d_s[:, None]

            # Calculate G = inv(I + A.T @ diag(1/psi_s) @ A)
            # This is part of the posterior mean calculation for z in Factor Analysis
            G = torch.linalg.inv(torch.eye(self.n_compo, device=device) + A.T@torch.diag(1/psi_s)@A)

            # E[z|x] = G @ A.T @ diag(1/psi_s) @ (x - d_s)
            E_z[:, stim_i*n_trial: (stim_i+1)*n_trial] = G@A.T@torch.diag(1/psi_s)@(x_s - d_s[:, None])

            # Reconstructed data: x_recon = d_s + A @ E[z|x]
            x_recon[:, stim_i*n_trial: (stim_i+1)*n_trial] = d_s[:, None] + A@E_z[:, stim_i*n_trial: (stim_i+1)*n_trial]

        return x_recon, E_z

    def train(self, lr0, x_test, n_trial_test):
        """
        Trains the model parameters with gradient descent to minimize the negative log likelihood.

        Args:
            lr0 (float): Initial learning rate for the Adam optimizer.
            x_test (np.ndarray): Test data for monitoring validation loss.
                                 Shape: (n_neurons, n_stimuli * n_trials_test).
            n_trial_test (int): Number of trials per stimulus in x_test.

        Returns:
            tuple:
                - self.d_p (torch.Tensor): Learned stimulus-specific mean responses.
                - self.alpha_p (torch.Tensor): Learned global loading matrix.
                - self.psi_p (torch.Tensor): Learned diagonal of private noise covariance.
        """
        # Initialize Adam optimizer for the learnable parameters (alpha_p, psi_p)
        optimizer = torch.optim.Adam([self.alpha_p, self.psi_p], lr=lr0)
        # Initialize Exponential learning rate scheduler
        decayRate = decayRate_general
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

        # Initialize old test NLL for early stopping
        NLL_old = self.loss_nll(x_test, n_trial_test)

        # Training loop
        for t in range(20001):  # Iterate up to 20000 steps
            optimizer.zero_grad()  # Zero the gradients before each backward pass

            # Calculate training and test NLL
            NLL = self.loss_nll(self.x, self.n_trial)
            NLL_test = self.loss_nll(x_test, n_trial_test)

            NLL.backward()  # Backpropagate to compute gradients
            optimizer.step()  # Update model parameters

            # Print progress and check for early stopping every 500 iterations
            if (t-1) % steps_print == 0:
                print(f"Iteration: {t}, Train Loss: {NLL.item():0.4f}, Test Loss: {NLL_test.item():0.4f}")
                # Early stopping condition: if test NLL starts increasing (or doesn't decrease significantly)
                if NLL_test > (NLL_old-tol):
                    print(f"Stop: Iteration: {t}, old test Loss: {NLL_old.item():0.5f}, new test Loss: {NLL_test.item():0.5f}")
                    break
                else:
                    NLL_old = NLL_test  # Update old test NLL
                    lr_scheduler.step()  # Step the learning rate scheduler
                    print('learning rate: ', lr_scheduler.get_last_lr())

        return self.d_p, self.alpha_p, self.psi_p


#################################
class affine_model():
    """
    Implements an affine model for neuronal population responses.
    This model extends the multiplicative model by adding a constant offset
    to the loading matrix.

    Model: x_st = d_s + A_s * z_st + epsilon_st
    where A_s = alpha_p * diag(d_s) + beta_p, with d_s being the stimulus-specific mean,
    alpha_p and beta_p being global loading matrices, z_st are latent variables, and
    epsilon_st is private noise.
    """

    def __init__(self,x, n, n_stim, n_trial, n_compo, alpha_p_init, beta_p_init, psi_p_init):
        """
        Initializes the affine_model.

        Args:
            x (np.ndarray): Input data with shape (n_neurons, n_stimuli * n_trials).
            n (int): Number of neurons.
            n_stim (int): Number of stimuli.
            n_trial (int): Number of trials per stimulus.
            n_compo (int): Number of latent components.
            alpha_p_init (np.ndarray): Initial guess for the multiplicative loading matrix 'alpha_p'.
                                       Shape: (n_neurons, n_compo).
            beta_p_init (np.ndarray): Initial guess for the additive loading matrix 'beta_p'.
                                      Shape: (n_neurons, n_compo).
            psi_p_init (np.ndarray): Initial guess for the diagonal of the
                                      private noise covariance 'psi_p'. Shape: (n_neurons, n_stimuli).
        """
        self.n = n
        self.n_stim = n_stim
        self.n_trial = n_trial
        self.x = x
        self.n_compo = n_compo
        self.SMALL = 1e-5

        # Initialize parameters similarly to multiplicative_model
        d_p = np.zeros((n, n_stim))
        alpha_p = alpha_p_init
        beta_p = beta_p_init
        psi_p = np.maximum(psi_p_init, self.SMALL)

        for stim_i in range(n_stim):
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)

        self.d_p = torch.from_numpy(d_p).to(device)

        self.alpha_p = torch.from_numpy(alpha_p).to(device)
        self.alpha_p.requires_grad=True

        self.beta_p = torch.from_numpy(beta_p).to(device)
        self.beta_p.requires_grad=True

        self.psi_p = torch.from_numpy(psi_p).to(device)
        self.psi_p.requires_grad=True

    def loss_nll(self, x, n_trial):
        """
        Calculates the negative log-likelihood (NLL) for the affine_model.
        The main difference from multiplicative_model is in the calculation of A.
        """
        x_var = torch.from_numpy(x).to(device)

        NLL = 0
        for stim_i in range(self.n_stim):
            d_s = self.d_p[:, stim_i]
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL, device=device))

            # Calculate the stimulus-specific loading matrix A
            # A = alpha_p * diag(d_s) + beta_p
            A = self.alpha_p*d_s[:,None] + self.beta_p
            cov = A@A.T + torch.diag(psi_s)

            to_learn = torch.distributions.multivariate_normal.MultivariateNormal(loc=d_s, covariance_matrix= cov)
            NLL += -torch.mean(to_learn.log_prob(x_var[:, stim_i*n_trial:(stim_i+1)*n_trial].T))

        return NLL/self.n_stim

    def recon_data(self, x, n_trial):
        """
        Reconstructs data and estimates latent variables for the affine_model.
        """
        x_var = torch.from_numpy(x).to(device)

        E_z = torch.zeros([self.n_compo, self.n_stim*n_trial], dtype=torch.double).to(device)
        x_recon = torch.zeros([self.n, self.n_stim*n_trial]).to(device)

        for stim_i in range(self.n_stim):
            d_s = self.d_p[:, stim_i]
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL, device=device))
            x_s = x_var[:, stim_i*n_trial:(stim_i+1)*n_trial]

            # Calculate the stimulus-specific loading matrix A
            A = self.alpha_p*d_s[:, None] + self.beta_p

            G = torch.linalg.inv(torch.eye(self.n_compo, device=device) + A.T@torch.diag(1/psi_s)@A)
            E_z[:, stim_i*n_trial: (stim_i+1)*n_trial] = G@A.T@torch.diag(1/psi_s)@(x_s - d_s[:, None])
            x_recon[:, stim_i*n_trial: (stim_i+1)*n_trial] = d_s[:, None] + A@E_z[:, stim_i*n_trial: (stim_i+1)*n_trial]

        return x_recon, E_z

    def train(self, lr0, x_test, n_trial_test):
        """
        Trains the affine_model parameters. Similar to multiplicative_model's train method,
        but includes `beta_p` in the optimizer.
        """
        optimizer = torch.optim.Adam([self.alpha_p, self.beta_p, self.psi_p], lr=lr0)
        decayRate = decayRate_general
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

        NLL_old = self.loss_nll(x_test, n_trial_test)

        for t in range(20001):
            optimizer.zero_grad()
            NLL = self.loss_nll(self.x, self.n_trial)
            NLL_test = self.loss_nll(x_test, n_trial_test)

            NLL.backward()
            optimizer.step()
            if (t-1) % steps_print == 0:
                print(f"Iteration: {t}, Train Loss: {NLL.item():0.4f}, Test Loss: {NLL_test.item():0.4f}")
                if NLL_test > (NLL_old-tol):
                    print(f"Stop: Iteration: {t}, old test Loss: {NLL_old.item():0.5f}, new test Loss: {NLL_test.item():0.5f}")
                    break
                else:
                    NLL_old = NLL_test
                    lr_scheduler.step()
                    print('learning rate: ', lr_scheduler.get_last_lr())

        return self.d_p, self.alpha_p, self.beta_p, self.psi_p


#####################################################
class additive_varp_model():
    """
    This is the additive model used in Xia et al 2023.
    This model assumes a global loading matrix (h_p) but stimulus-specific
    private noise variance (psi_p).

    Model: x_st = d_s + h_p * z_st + epsilon_st
    where h_p is a global loading matrix, z_st are latent variables, and
    epsilon_st is private noise with stimulus-specific variance (psi_s).
    """

    def __init__(self,x, n, n_stim, n_trial, n_compo, h_p_init, psi_p_init):
        """
        Initializes the additive_varp_model.

        Args:
            x (np.ndarray): Input data with shape (n_neurons, n_stimuli * n_trials).
            n (int): Number of neurons.
            n_stim (int): Number of stimuli.
            n_trial (int): Number of trials per stimulus.
            n_compo (int): Number of latent components.
            h_p_init (np.ndarray): Initial guess for the global loading matrix 'h_p'.
                                   Shape: (n_neurons, n_compo).
            psi_p_init (np.ndarray): Initial guess for the diagonal of the
                                      private noise covariance 'psi_p'. Shape: (n_neurons, n_stimuli).
        """
        self.n = n
        self.n_stim = n_stim
        self.n_trial = n_trial
        self.x = x
        self.n_compo = n_compo
        self.SMALL = 1e-5

        # Initialize parameters
        d_p = np.zeros((n, n_stim))
        # Calculate initial stimulus-specific mean responses
        for stim_i in range(n_stim):
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)

        h_p = h_p_init
        psi_p = np.maximum(psi_p_init, self.SMALL)

        self.d_p = torch.from_numpy(d_p).to(device)

        self.h_p = torch.from_numpy(h_p).to(device)
        self.h_p.requires_grad=True

        self.psi_p = torch.from_numpy(psi_p).to(device)
        self.psi_p.requires_grad=True

    def loss_nll(self, x, n_trial):
        """
        Calculates the negative log-likelihood (NLL) for the additive_varp_model.
        Here, the loading matrix 'A' is simply 'h_p'.
        """
        x_var = torch.from_numpy(x).to(device)

        NLL = 0
        for stim_i in range(self.n_stim):
            d_s = self.d_p[:, stim_i]
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL, device=device))

            # The loading matrix A is simply h_p (global factor)
            A = self.h_p
            cov = A@A.T + torch.diag(psi_s)

            to_learn = torch.distributions.multivariate_normal.MultivariateNormal(loc=d_s, covariance_matrix= cov)
            NLL += -torch.mean(to_learn.log_prob(x_var[:, stim_i*n_trial:(stim_i+1)*n_trial].T))

        return NLL/self.n_stim

    def recon_data(self, x, n_trial):
        """
        Reconstructs data and estimates latent variables for the additive_varp_model.
        """
        x_var = torch.from_numpy(x).to(device)

        E_z = torch.zeros([self.n_compo, self.n_stim*n_trial], dtype=torch.double).to(device)
        x_recon = torch.zeros([self.n, self.n_stim*n_trial]).to(device)

        for stim_i in range(self.n_stim):
            d_s = self.d_p[:, stim_i]
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL, device=device))
            x_s = x_var[:, stim_i*n_trial:(stim_i+1)*n_trial]

            A = self.h_p  # Loading matrix is global h_p

            G = torch.linalg.inv(torch.eye(self.n_compo, device=device) + A.T@torch.diag(1/psi_s)@A)
            E_z[:, stim_i*n_trial: (stim_i+1)*n_trial] = G@A.T@torch.diag(1/psi_s)@(x_s - d_s[:, None])
            x_recon[:, stim_i*n_trial: (stim_i+1)*n_trial] = d_s[:, None] + A@E_z[:, stim_i*n_trial: (stim_i+1)*n_trial]

        return x_recon, E_z

    def train(self, lr0, x_test, n_trial_test):
        """
        Trains the additive_varp_model parameters.
        """
        optimizer = torch.optim.Adam([self.h_p, self.psi_p], lr=lr0)
        decayRate = decayRate_general
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

        NLL_old = self.loss_nll(x_test, n_trial_test)

        for t in range(20001):
            optimizer.zero_grad()
            NLL = self.loss_nll(self.x, self.n_trial)
            NLL_test = self.loss_nll(x_test, n_trial_test)

            NLL.backward()
            optimizer.step()
            if (t-1) % steps_print == 0:
                print(f"Iteration: {t}, Train Loss: {NLL.item():0.4f}, Test Loss: {NLL_test.item():0.4f}")
                if NLL_test > (NLL_old-tol):
                    print(f"Stop: Iteration: {t}, old test Loss: {NLL_old.item():0.5f}, new test Loss: {NLL_test.item():0.5f}")
                    break
                else:
                    NLL_old = NLL_test
                    lr_scheduler.step()
                    print('learning rate: ', lr_scheduler.get_last_lr())

        return self.d_p, self.h_p, self.psi_p


##########################################################
class generalized_model():
    """
    Implements a generalized Factor Analysis (FA) model.
    This model fits a separate Factor Analysis for each stimulus.
    It assumes both shared (latent) and private (noise) variability
    are stimulus-dependent.

    Model: x_st = d_s + F_s * z_st + epsilon_st
    where F_s is a stimulus-specific loading matrix, z_st are latent variables,
    and epsilon_st is private noise with stimulus-specific variance (psi_s).
    """

    def __init__(self,x, n, n_stim, n_trial, x_test, n_trial_test, n_compo):
        """
        Initializes the generalized_model. This model does not use PyTorch
        for training; instead, it leverages `sklearn.decomposition.FactorAnalysis`
        directly for each stimulus.

        Args:
            x (np.ndarray): Training data with shape (n_neurons, n_stimuli * n_trials).
            n (int): Number of neurons.
            n_stim (int): Number of stimuli.
            n_trial (int): Number of trials per stimulus in training data.
            x_test (np.ndarray): Test data with shape (n_neurons, n_stimuli * n_trials_test).
            n_trial_test (int): Number of trials per stimulus in test data.
            n_compo (int): Number of latent components.
        """
        # Initialize parameters and storage for results
        d_p = np.zeros((n, n_stim))  # Stimulus-specific mean
        psi_p = np.ones((n, n_stim))  # Stimulus-specific private noise variance
        F_p = np.zeros((n, n_compo, n_stim))  # Stimulus-specific loading matrix

        z = np.zeros((n_stim*n_trial, n_compo))  # Latent variables for training data
        z_test = np.zeros((n_stim*n_trial_test, n_compo))  # Latent variables for test data

        x_recon = np.zeros_like(x, dtype=float)  # Reconstructed training data
        x_test_recon = np.zeros_like(x_test, dtype=float)  # Reconstructed test data

        ll = 0  # Log-likelihood for training data
        ll_test = 0  # Log-likelihood for test data

        # Iterate through each stimulus to fit a separate Factor Analysis model
        for stim_i in range(n_stim):
            # Extract training data for the current stimulus
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            # Calculate stimulus-specific mean
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)

            # Subtract the mean to get residuals (zero-mean data for FA)
            res_x = x_tmp - d_p[:, stim_i:stim_i+1]
            fa = FactorAnalysis()
            fa.n_components = n_compo
            fa.fit(res_x.T)  # Fit FA to the transposed residuals (samples as rows)

            # Store the learned FA parameters
            F_p[:,:, stim_i] = fa.components_.T  # components_ is (n_compo, n_features), transpose to (n_features, n_compo)
            psi_p[:,stim_i] = fa.noise_variance_

            # Calculate log-likelihood for training data
            ll += fa.score(res_x.T)

            # Process test data for the current stimulus
            x_test_tmp = x_test[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test]
            res_x_test = x_test_tmp - d_p[:, stim_i:stim_i+1]

            # Calculate log-likelihood for test data
            ll_test += fa.score(res_x_test.T)

            # Transform data to latent space (estimate z)
            z[stim_i*n_trial:(stim_i+1)*n_trial, :] = fa.transform(res_x.T)
            z_test[stim_i*n_trial_test:(stim_i+1)*n_trial_test, :] = fa.transform(res_x_test.T)

            # Reconstruct data
            x_recon[:, stim_i*n_trial:(stim_i+1)*n_trial] = d_p[:, stim_i:stim_i+1] + F_p[:,:, stim_i] @ (z[stim_i*n_trial:(stim_i+1)*n_trial, :].T)
            x_test_recon[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test] = d_p[:, stim_i:stim_i+1] + F_p[:,:,stim_i] @ (z_test[stim_i*n_trial_test:(stim_i+1)*n_trial_test, :].T)

        # Store all learned parameters and calculated metrics
        self.d_p = d_p
        self.psi_p = psi_p
        self.F_p = F_p
        self.z = z
        self.z_test = z_test

        # Convert log-likelihood to negative log-likelihood (NLL)
        self.NLL = -ll/n_stim
        self.NLL_test = -ll_test/n_stim

        print('test NLL: ', self.NLL_test, 'train NLL: ', self.NLL)

        self.x_recon = x_recon
        self.x_test_recon = x_test_recon


###################################################
class additive_model():
    """
    This is NOT the additive model used in Xia et al.
    This additive model assumes stimulus-independent private variability for each neuron
    and a global loading matrix.

    Model: x_st = d_s + h_p * z_st + epsilon_t
    where h_p is a global loading matrix, z_st are latent variables, and
    epsilon_t is private noise with global (stimulus-independent) variance (psi_p).
    """

    def __init__(self,x, n, n_stim, n_trial, x_test, n_trial_test, n_compo):
        """
        Initializes the additive_model. Similar to generalized_model, this uses
        `sklearn.decomposition.FactorAnalysis` on the entire dataset of residuals.

        Args:
            x (np.ndarray): Training data with shape (n_neurons, n_stimuli * n_trials).
            n (int): Number of neurons.
            n_stim (int): Number of stimuli.
            n_trial (int): Number of trials per stimulus in training data.
            x_test (np.ndarray): Test data with shape (n_neurons, n_stimuli * n_trials_test).
            n_trial_test (int): Number of trials per stimulus in test data.
            n_compo (int): Number of latent components.
        """
        # Initialize parameters and storage
        d_p = np.zeros((n, n_stim))

        z = np.zeros((n_stim*n_trial, n_compo))
        z_test = np.zeros((n_stim*n_trial_test, n_compo))

        x_recon = np.zeros_like(x, dtype=float)
        x_test_recon = np.zeros_like(x_test, dtype=float)

        ll = 0
        ll_test = 0

        res_x  = np.zeros_like(x)  # Residuals for training data
        res_x_test = np.zeros_like(x_test)  # Residuals for test data

        # Calculate stimulus-specific means and residuals for both train and test data
        for stim_i in range(n_stim):
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)
            res_x[:, stim_i*n_trial:(stim_i+1)*n_trial] = x_tmp - d_p[:, stim_i:stim_i+1]

            x_test_tmp = x_test[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test]
            res_x_test[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test] = x_test_tmp - d_p[:, stim_i:stim_i+1]

        # Fit a single Factor Analysis model to all training residuals combined
        fa = FactorAnalysis()
        fa.n_components = n_compo
        fa.fit(res_x.T)  # Fit to the transposed residuals (all trials combined)

        # h_p is the global loading matrix, psi_p is the global private noise variance
        h_p = fa.components_.T # h_p is n_neurons x n_compo
        psi_p = fa.noise_variance_

        # Calculate log-likelihood for all combined training and test residuals
        ll = fa.score(res_x.T)
        ll_test = fa.score(res_x_test.T)

        # Transform all data to latent space
        z = fa.transform(res_x.T) # z is n_samples x n_compo
        z_test = fa.transform(res_x_test.T)

        # Reconstruct data
        for stim_i in range(n_stim):
            x_recon[:, stim_i*n_trial:(stim_i+1)*n_trial] = d_p[:, stim_i:stim_i+1] + h_p @ (z[stim_i*n_trial:(stim_i+1)*n_trial, :].T)
            x_test_recon[:, stim_i*n_trial_test:(stim_i+1)*n_trial_test] = d_p[:, stim_i:stim_i+1] + h_p @ (z_test[stim_i*n_trial_test:(stim_i+1)*n_trial_test, :].T)

        # Store results
        self.d_p = d_p
        self.psi_p = psi_p
        self.h_p = h_p
        self.z = z
        self.z_test = z_test

        self.NLL = -ll
        self.NLL_test = -ll_test

        print('test NLL: ', self.NLL_test, 'train NLL: ', self.NLL)

        self.x_recon = x_recon
        self.x_test_recon = x_test_recon


class exponent_model():
    """
    This model introduces an exponent parameter to the multiplicative component
    of the loading matrix. This model is not used in Xia et al 2023.

    Model: x_st = d_s + A_s * z_st + epsilon_st
    where A_s = alpha_p * (d_s^expo_p) + beta_p, with d_s being the stimulus-specific mean,
    alpha_p and beta_p being global loading matrices, expo_p is a global exponent,
    z_st are latent variables, and epsilon_st is private noise.
    """

    def __init__(self,x, n, n_stim, n_trial, n_compo, expo_p_init, beta_p_init, psi_p_init):
        """
        Initializes the exponent_model.

        Args:
            x (np.ndarray): Input data with shape (n_neurons, n_stimuli * n_trials).
            n (int): Number of neurons.
            n_stim (int): Number of stimuli.
            n_trial (int): Number of trials per stimulus.
            n_compo (int): Number of latent components.
            expo_p_init (float): Initial guess for the exponent parameter.
            beta_p_init (np.ndarray): Initial guess for the additive loading matrix 'beta_p'.
                                      Shape: (n_neurons, n_compo).
            psi_p_init (np.ndarray): Initial guess for the diagonal of the
                                      private noise covariance 'psi_p'. Shape: (n_neurons, n_stimuli).
        """
        self.n = n
        self.n_stim = n_stim
        self.n_trial = n_trial
        self.x = x
        self.n_compo = n_compo
        self.SMALL = 1e-5

        # Initialize parameters
        d_p = np.zeros((n, n_stim))
        # Initialized to zeros, consider more robust initialization for alpha_p if needed.
        alpha_p = np.zeros((n,n_compo))
        beta_p = beta_p_init
        psi_p = np.maximum(psi_p_init, self.SMALL)
        expo_p = expo_p_init

        for stim_i in range(n_stim):
            x_tmp = x[:, stim_i*n_trial:(stim_i+1)*n_trial]
            d_p[:, stim_i] = np.mean(x_tmp, axis=1)

        self.d_p = torch.from_numpy(d_p).to(device)

        self.alpha_p = torch.from_numpy(alpha_p).to(device)
        self.alpha_p.requires_grad=True

        self.beta_p = torch.from_numpy(beta_p).to(device)
        self.beta_p.requires_grad=True

        self.psi_p = torch.from_numpy(psi_p).to(device)
        self.psi_p.requires_grad=True

        # Ensure float64 for exponentiation stability during gradient descent.
        self.expo_p = torch.tensor(expo_p, device=device, dtype=torch.float64)
        self.expo_p.requires_grad=True

    def loss_nll(self, x, n_trial):
        """
        Calculates the negative log-likelihood (NLL) for the exponent_model.
        The key difference is the term `d_s[:,None]**self.expo_p` in the A matrix.
        """
        x_var = torch.from_numpy(x).to(device)

        NLL = 0
        for stim_i in range(self.n_stim):
            d_s = self.d_p[:, stim_i]
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL, device=device))

            # Calculate the stimulus-specific loading matrix A with exponent
            # A = alpha_p * (d_s^expo_p) + beta_p
            A = self.alpha_p*(d_s[:,None]**self.expo_p) + self.beta_p
            cov = A@A.T + torch.diag(psi_s)

            to_learn = torch.distributions.multivariate_normal.MultivariateNormal(loc=d_s, covariance_matrix= cov)
            NLL += -torch.mean(to_learn.log_prob(x_var[:, stim_i*n_trial:(stim_i+1)*n_trial].T))

        return NLL/self.n_stim

    def recon_data(self, x, n_trial):
        """
        Reconstructs data and estimates latent variables for the exponent_model.
        """
        x_var = torch.from_numpy(x).to(device)

        E_z = torch.zeros([self.n_compo, self.n_stim*n_trial], dtype=torch.double).to(device)
        x_recon = torch.zeros([self.n, self.n_stim*n_trial]).to(device)

        for stim_i in range(self.n_stim):
            d_s = self.d_p[:, stim_i]
            psi_s = torch.maximum(self.psi_p[:, stim_i], torch.tensor(self.SMALL, device=device))
            x_s = x_var[:, stim_i*n_trial:(stim_i+1)*n_trial]

            A = self.alpha_p*(d_s[:, None]**self.expo_p) + self.beta_p

            G = torch.linalg.inv(torch.eye(self.n_compo, device=device) + A.T@torch.diag(1/psi_s)@A)
            E_z[:, stim_i*n_trial: (stim_i+1)*n_trial] = G@A.T@torch.diag(1/psi_s)@(x_s - d_s[:, None])
            x_recon[:, stim_i*n_trial: (stim_i+1)*n_trial] = d_s[:, None] + A@E_z[:, stim_i*n_trial: (stim_i+1)*n_trial]

        return x_recon, E_z

    def train(self, lr0, x_test, n_trial_test):
        """
        Trains the exponent_model parameters. Includes `expo_p` in the optimizer.
        """
        optimizer = torch.optim.Adam([self.alpha_p, self.beta_p, self.psi_p, self.expo_p], lr=lr0)
        decayRate = decayRate_general
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)

        NLL_old = self.loss_nll(x_test, n_trial_test)

        for t in range(20001):
            optimizer.zero_grad()
            NLL = self.loss_nll(self.x, self.n_trial)
            NLL_test = self.loss_nll(x_test, n_trial_test)

            NLL.backward()
            optimizer.step()
            if (t-1) % steps_print == 0:
                print(f"Iteration: {t}, Train Loss: {NLL.item():0.4f}, Test Loss: {NLL_test.item():0.4f}")
                if NLL_test > (NLL_old-tol):
                    print(f"Stop: Iteration: {t}, old test Loss: {NLL_old.item():0.5f}, new test Loss: {NLL_test.item():0.5f}")
                    break
                else:
                    NLL_old = NLL_test
                    lr_scheduler.step()
                    print('learning rate: ', lr_scheduler.get_last_lr())

        return self.d_p, self.alpha_p, self.beta_p, self.psi_p, self.expo_p