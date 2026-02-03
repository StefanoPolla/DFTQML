"""
Tests for AE autoencoder with both standard and linear encoder modes.
"""
import torch
import pytest
from dftqml.autoencoders.ae import AE

@pytest.mark.parametrize("linear_encoder", [False, True])
def test_ae_forward_shapes(linear_encoder):
    # Setup: 2D input, batch size 8
    batch_size = 8
    input_shape = (3, 4)
    input_dim = input_shape
    hidden_dims = [10, 6]
    latent_dim = 5
    model = AE(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        use_lipschitz_layers=False,
        use_batchnorm=False,
        linear_encoder=linear_encoder,
    )
    dummy_data = torch.randn(batch_size, *input_shape)
    model.initialize_normalization(dummy_data)
    # Forward pass
    z = model.encode(dummy_data)
    x_rec = model.decode(z)
    # Check shapes
    assert z.shape == (batch_size, latent_dim)
    assert x_rec.shape == (batch_size, *input_shape)

@pytest.mark.parametrize("linear_encoder", [False, True])
def test_ae_training_step(linear_encoder):
    # Setup: 1D input, batch size 16
    batch_size = 16
    input_dim = 12
    hidden_dims = [20, 10]
    latent_dim = 4
    model = AE(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        use_lipschitz_layers=False,
        use_batchnorm=True,
        linear_encoder=linear_encoder,
    )
    dummy_data = torch.randn(batch_size, input_dim)
    model.initialize_normalization(dummy_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    # One training step
    model.train()
    optimizer.zero_grad()
    z = model.encode(dummy_data)
    x_rec = model.decode(z)
    loss = torch.nn.functional.mse_loss(x_rec, dummy_data)
    loss.backward()
    optimizer.step()
    # Check that loss is a scalar and gradients are updated
    assert loss.item() > 0
    for p in model.parameters():
        if p.grad is not None:
            assert torch.any(p.grad != 0)

def test_ae_linear_encoder_lipschitz_error():
    # Linear encoder: encoder Lipschitz bound should be zero, decoder bound should be positive
    input_dim = 8
    hidden_dims = [10, 6]
    latent_dim = 3
    model = AE(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        use_lipschitz_layers=True,
        linear_encoder=True,
    )
    encoder_bound, decoder_bound = model.log_lipschitz_bounds
    # Encoder bound should be a zero tensor
    assert isinstance(encoder_bound, torch.Tensor)
    assert encoder_bound.numel() == 1
    assert encoder_bound.item() == 0.0
    # Decoder bound should be positive
    assert isinstance(decoder_bound, torch.Tensor)
    assert decoder_bound.numel() == 1
    assert decoder_bound.item() > 0.0
