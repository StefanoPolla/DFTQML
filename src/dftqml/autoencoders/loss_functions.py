import torch
import torch.nn.functional as F


def _loss_inFidelity(input_state, reconstructed_state, reduction_type="mean"):
    # Compute inner product for each batch item
    overlap = torch.sum(input_state * reconstructed_state, dim=1)
    fidelity = torch.abs(overlap)
    infidelity = 1 - fidelity

    if reduction_type == "sum":
        return torch.sum(infidelity)
    elif reduction_type == "mean":
        return torch.mean(infidelity)
    elif reduction_type is None or reduction_type == "none":
        return infidelity
    else:
        raise ValueError(f"Invalid reduction_type: {reduction_type}")


def _loss_norm_statevector(reconstructed_state, reduction_type="mean"):
    # Compute norm per sample (batch)

    if reconstructed_state.dim() > 2:
        reconstructed_state = reconstructed_state.flatten(start_dim=1)

    norm = torch.sqrt(torch.sum(reconstructed_state**2, dim=1))
    loss = (1 - norm) ** 2

    if reduction_type == "sum":
        return torch.sum(loss)
    elif reduction_type == "mean":
        return torch.mean(loss)
    elif reduction_type is None or reduction_type == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction_type: {reduction_type}")


def _loss_norm_latent(latent_vector, reduction_type="mean"):
    norm_sq = latent_vector**2

    if reduction_type == "sum":
        loss = torch.sum(norm_sq)
    elif reduction_type == "mean":
        loss = torch.mean(norm_sq)
    elif reduction_type is None or reduction_type == "none":
        loss = norm_sq
    else:
        raise ValueError(f"Invalid reduction_type: {reduction_type}")

    return loss


def _loss_norm_latent_well(latent_vector, well_radius=2.0, reduction_type="mean"):
    """
    Compute a well-based loss for the latent space.

    The loss is 0 for latent norms within a radius, and increases quadratically outside the well.

    Args:
        latent_vector: The latent representation tensor
        well_radius: The radius of the well (default 1.0 for [-1, 1] range)
        reduction_type: Type of reduction to apply

    Returns:
        The well-based latent loss
    """
    # Compute norm of the latent vector
    norm = torch.norm(latent_vector, p=2, dim=1)

    # subtract the well radius and apply ReLU to clamp negative values to zero
    loss = torch.relu(norm - well_radius)

    # Apply reduction
    if reduction_type == "sum":
        loss = torch.sum(loss)
    elif reduction_type == "mean":
        loss = torch.mean(loss)
    elif reduction_type is None or reduction_type == "none":
        loss = loss
    else:
        raise ValueError(f"Invalid reduction_type: {reduction_type}")

    return loss


def _loss_latent_contrastive_repulsion(
    latent_vector, input_vector, epsilon=1e-6, reduction_type="mean"
):
    """
    Computes a contrastive repulsion loss between input and latent representations.

    This loss encourages the pairwise distances in the latent space to reflect those in the input space,
    by penalizing cases where points that are far apart in the input space are mapped too close together
    in the latent space. The loss is computed as the ratio of squared input distances to squared latent
    distances (plus a small epsilon for numerical stability), averaged or summed over all unique pairs.

    Args:
        latent_vector: Latent representations of shape (batch_size, latent_dim).
        input_vector: Input representations of shape (batch_size, input_dim).
        epsilon: Small value to avoid division by zero. Default is 1e-6.
        reduction_type: Specifies the reduction to apply to the output: "mean" | "sum".

    Returns:
        torch.Tensor: The computed contrastive repulsion loss (scalar).
    """

    if input_vector.dim() > 2:
        input_vector = input_vector.flatten(start_dim=1)

    input_distances = torch.cdist(input_vector, input_vector, p=2)
    latent_distances = torch.cdist(latent_vector, latent_vector, p=2)

    # Exclude self-pairs (i = j)
    mask = torch.triu(torch.ones_like(input_distances, dtype=torch.bool), diagonal=1)

    repulsion_terms = (input_distances[mask] ** 2) / (
        latent_distances[mask] ** 2 + epsilon
    )

    # normalize by average input distance
    avg_input_distance = (input_distances[mask] ** 2).mean().item()
    repulsion_terms /= avg_input_distance

    # Apply reduction
    if reduction_type == "sum":
        loss = torch.sum(repulsion_terms)
    elif reduction_type == "mean":
        loss = torch.mean(repulsion_terms)
    else:
        raise ValueError(f"Invalid reduction_type: {reduction_type}")

    return loss


def _loss_latent_repulsion(latent_vector, latent_epsilon=1e-8, reduction_type="mean"):
    # Compute pairwise distances (L2)
    pairwise_distances = torch.cdist(latent_vector, latent_vector, p=2)

    # Exclude self-pairs (i = j)
    mask = torch.triu(torch.ones_like(pairwise_distances, dtype=torch.bool), diagonal=1)
    repulsion_terms = 1.0 / (pairwise_distances[mask] ** 2 + latent_epsilon)

    # Apply reduction
    if reduction_type == "sum":
        loss = torch.sum(repulsion_terms)
    elif reduction_type == "mean":
        loss = torch.mean(repulsion_terms)
    elif reduction_type is None or reduction_type == "none":
        loss = repulsion_terms
    else:
        raise ValueError(f"Invalid reduction_type: {reduction_type}")

    return loss


def loss_function(
    input_vector,
    reconstructed_vector,
    latent_vector,
    log_lipschitz_bounds=None,
    lambda_latent_norm=0.0,
    lambda_latent_repulsion=0.0,
    lambda_lipschitz_encoder=0.0,
    lambda_lipschitz_decoder=0.0,
    epsilon_latent_repulsion=1e-8,
    reduction_type="mean",
    use_statevector=False,
    return_components=False,
    use_well_loss=True,
    use_contrastive_repulsion=True,
):
    """
    Compute the total loss for the autoencoder, including reconstruction loss, latent space regularization,
    latent repulsion, and optional Lipschitz regularization.

    Args:
        input_vector (torch.Tensor): Original input data.
        reconstructed_vector (torch.Tensor): Reconstructed data from the autoencoder.
        latent_vector (torch.Tensor): Encoded (latent) representation of the input data.
        lipschitz_bound (tuple or None): Lipschitz regularization losses for encoder and decoder (default None).
        lambda_latent_norm (float): Regularization coefficient for the latent norm loss (default 0.0).
        lambda_latent_repulsion (float): Coefficient for the latent repulsion loss (default 0.0).
        lambda_lipschitz (float): Coefficient for the Lipschitz regularization loss (default 0.0).
        epsilon_latent_repulsion (float): Small constant to avoid division by zero in repulsion loss (default 1e-8).
        reduction_type (str): Reduction to apply to all loss components ("mean", "sum", or "none").
        use_statevector (bool): If True, use statevector loss (infidelity + normalization) for reconstruction.
        return_components (bool): If True, return individual loss components.
        use_well_loss (bool): If True, use well-based latent norm loss.
        use_contrastive_repulsion (bool): If True, use contrastive repulsion loss.

    Returns:
        torch.Tensor: Total loss for the autoencoder.
        or
        tuple: (total_loss, reconstruction_loss, latent_norm_loss, latent_repulsion_loss, lipschitz_bound[0], lipschitz_bound[1])
            if return_components is True.
    """
    # Reconstruction loss
    if use_statevector:
        reconstruction_loss = _loss_inFidelity(
            input_vector, reconstructed_vector, reduction_type=reduction_type
        ) + _loss_norm_statevector(reconstructed_vector, reduction_type=reduction_type)
    else:
        reconstruction_loss = F.mse_loss(
            input_vector, reconstructed_vector, reduction=reduction_type
        )
    total_loss = reconstruction_loss.clone()

    if lambda_latent_norm != 0.0 or return_components:
        if use_well_loss:
            latent_norm_loss = _loss_norm_latent_well(
                latent_vector, reduction_type=reduction_type
            )
        else:
            latent_norm_loss = _loss_norm_latent(
                latent_vector, reduction_type=reduction_type
            )
        total_loss += lambda_latent_norm * latent_norm_loss

    if lambda_latent_repulsion != 0.0 or return_components:
        if use_contrastive_repulsion:
            # Contrastive repulsion loss
            latent_repulsion_loss = _loss_latent_contrastive_repulsion(
                latent_vector,
                input_vector,
                epsilon=epsilon_latent_repulsion,
                reduction_type=reduction_type,
            )
        else:
            latent_repulsion_loss = _loss_latent_repulsion(
                latent_vector,
                latent_epsilon=epsilon_latent_repulsion,
                reduction_type=reduction_type,
            )
        total_loss += lambda_latent_repulsion * latent_repulsion_loss

    if lambda_lipschitz_encoder > 0 or lambda_lipschitz_decoder > 0:
        if log_lipschitz_bounds is None:
            raise ValueError(
                "log_lipschitz_bounds must be provided if lambda_lipschitz_encoder or lambda_lipschitz_decoder > 0"
            )
        lipschitz_encoder_loss = log_lipschitz_bounds[0]
        lipschitz_decoder_loss = log_lipschitz_bounds[1]
        total_loss += lambda_lipschitz_encoder * lipschitz_encoder_loss
        total_loss += lambda_lipschitz_decoder * lipschitz_decoder_loss

    if return_components and log_lipschitz_bounds is not None:
        return (
            total_loss,
            reconstruction_loss,
            latent_norm_loss,
            latent_repulsion_loss,
            log_lipschitz_bounds[0],
            log_lipschitz_bounds[1],
        )
    elif return_components:
        return total_loss, reconstruction_loss, latent_norm_loss, latent_repulsion_loss
    else:
        return total_loss
