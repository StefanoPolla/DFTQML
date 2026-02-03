import time
import torch
from dftqml.autoencoders.ae import load_model
from dftqml.hubbard_model.fhchain import FermiHubbardChain
import json


class LatentEnergyOptimizer:
    def __init__(self, checkpoint_dir, system_params, device="cpu"):
        """
        Args:
            checkpoint_dir (str): Path to trained AE checkpoint.
            system_params (dict): Dict with keys n_sites, n_particles, u, t, etc.
            device (str): "cpu" or "cuda"
        """
        self.device = device
        self.model = load_model(checkpoint_dir, device=device)
        self.system = FermiHubbardChain(**system_params)

        with open(f"{checkpoint_dir}/model_config.json", "r") as f:
            model_config = json.load(f)
        if len(model_config["input_shape"]) == 4:
            self.observable_type = "two_rdm"
            self._energy_from_observable = self.system.ground_energy_from_rdm
        elif len(model_config["input_shape"]) == 2:
            self.observable_type = "ham_terms"
            self._energy_from_observable = (
                self.system.ground_energy_from_hamiltonian_terms
            )
        else:
            raise ValueError("Unknown input shape in model_config")

    def energy_from_latent(self, z, potential=None):
        observable = self.model.decode(z.unsqueeze(0)).squeeze(0)
        energy = self._energy_from_observable(observable, potential)
        return energy

    def minimize(
        self,
        potential,
        *,
        init_z=None,
        opt=None,
        lr=1e-2,
        max_iter=100,
        energy_tol=1e-6,
        grad_tol=1e-4,
        well_size=2.0,
        well_penalty_strength=50.0,
        verbose=False,
        print_interval=1,
        use_grad_tol=False,
        return_loss_history=False,
        return_grad_history=False,
    ):
        """
        Args:
            potential (np.ndarray): Potential array, shape (L, L)
            init_z (np.ndarray): Initial latent vector, shape (latent_dim,)
            opt (torch.optim.Optimizer): Optimizer class. If not set, defaults to Adam
            lr (float): Learning rate for the optimizer
            max_iter (int): Maximum number of iterations for optimization
            energy_tol (float): Tolerance for energy convergence
            well_size (float): Size of the well for the penalty term
            well_penalty_strength (float): Strength of the penalty term
            verbose (bool): Whether to print progress
            print_interval (int): Interval for printing progress
            return_loss_history (bool): Whether to return the loss history


        Returns:
            z: Optimized latent vector, shape (1, latent_dim)
            energy: Optimized energy value
            loss_history: List of loss values during optimization
        """
        # print all arguments if verbose
        if verbose:
            print("Minimization parameters:")
            print(f"  Learning rate: {lr}")
            print(f"  Max iterations: {max_iter}")
            print(f"  Energy tolerance: {energy_tol}")
            print(f"  Well size: {well_size}")
            print(f"  Well penalty strength: {well_penalty_strength}")
            print(f"  Print interval: {print_interval}")
        loss_history = []
        grad_history = []

        # Initialize latent vector
        latent_dim = self.model.decoder[0].in_features
        if init_z is None:
            z = torch.zeros((1, latent_dim), requires_grad=True, device=self.device)
        else:
            z = torch.tensor(
                init_z, dtype=torch.float32, requires_grad=True, device=self.device
            )

        if opt is None:
            optimizer = torch.optim.Adam([z], lr=lr)
        else:
            optimizer = opt([z], lr=lr)

        def cost_fn(z_vec):
            base_loss = self.energy_from_latent(z_vec, potential)
            if well_penalty_strength > 0:
                norm = z_vec.norm(p=2)
                penalty = well_penalty_strength * torch.relu(norm - well_size) ** 2
                return base_loss + penalty
            return base_loss

        def closure():
            optimizer.zero_grad()
            loss = cost_fn(z)
            loss.backward()
            loss_history.append(loss.item())
            grad_history.append(z.grad.norm().item())
            return loss

        # Compute and print the initial cost
        initial_cost = cost_fn(z)
        if verbose:
            print(f"Initial cost: {initial_cost}")

        cost_old = float("inf")
        start = time.time()

        # ***** Optimization loop *****

        for step in range(max_iter):
            loss_tensor = optimizer.step(closure)
            cost = loss_tensor.item()

            delta = abs(cost - cost_old)

            # Logging every print_interval steps
            if verbose:
                if step % print_interval == 0:
                    print(f"Step {step}: cost = {cost:.10f}")
                    print(f"Gradient norm = {z.grad.norm().item():.10f}")
                    if step > 0:
                        print(f"Delta cost = {delta:.10f}")
                    else:
                        print("Delta cost = N/A (first recorded step)")

            # Check for convergence

            if use_grad_tol:
                if step > 2 and z.grad.norm().item() < grad_tol:
                    if verbose:
                        print(f"Gradient convergence reached at step {step}.")
                    break
            elif step > 2 and delta < energy_tol:
                if verbose:
                    print(f"Energy convergence reached at step {step}.")
                break

            cost_old = cost

        if verbose:
            elapsed_time = time.time() - start
            print("\nOptimization complete.")
            print(f"Optimization took {elapsed_time:.2f} seconds.")
            print(f"Final optimized cost: {cost:.10f}")

        opt_energy = self.energy_from_latent(z, potential).detach().item()
        # Move to CPU before converting to NumPy (GPU-safe)
        z = z.detach().cpu().numpy()

        if return_loss_history and return_grad_history:
            return z, opt_energy, loss_history, grad_history
        elif return_loss_history:
            return z, opt_energy, loss_history
        else:
            return z, opt_energy


# def torch_inverse_transform(x, mean, scale, flatten=True, original_shape=None):
#     """
#     PyTorch differentiable inverse_transform for StandardScaler + flattening.

#     Args:
#         x (torch.Tensor): Input tensor, shape (batch, features)
#         mean (array-like or torch.Tensor): Mean used for normalization (shape: (features,))
#         scale (array-like or torch.Tensor): Scale used for normalization (shape: (features,))
#         flatten (bool): Whether the data was flattened during preprocessing
#         original_shape (tuple): The original shape before flattening (e.g., (batch, ...))

#     Returns:
#         torch.Tensor: Inverse transformed tensor, shape (batch, ...) if flatten else (batch, features)
#     """
#     # Ensure mean and scale are torch tensors on the same device/dtype as x
#     if not torch.is_tensor(mean):
#         mean = torch.tensor(mean, dtype=x.dtype, device=x.device)
#     if not torch.is_tensor(scale):
#         scale = torch.tensor(scale, dtype=x.dtype, device=x.device)

#     # Inverse transform: x_orig = x_scaled * scale + mean
#     x_orig = x * scale + mean

#     # Reshape if needed
#     if flatten and original_shape is not None:
#         x_orig = x_orig.view(-1, *original_shape[1:])

#     return x_orig
