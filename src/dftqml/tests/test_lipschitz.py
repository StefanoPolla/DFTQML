"""
Tests for Lipschitz-constrained layers and regularization.

[AI generated tests with GitHub copilot, Claude Sonnet 4]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pytest

from dftqml.autoencoders.lipschitz import LipschitzLinear
from dftqml.autoencoders.ae import AE
from dftqml.autoencoders.loss_functions import loss_function


class TestLipschitzLinear:
    """Test the LipschitzLinear layer implementation."""

    def test_lipschitz_linear_forward(self):
        """Test that LipschitzLinear layer performs forward pass correctly."""
        layer = LipschitzLinear(4, 2)
        input_data = torch.randn(5, 4)
        output = layer(input_data)
        
        assert output.shape == (5, 2)
        assert output.requires_grad
        
    def test_lipschitz_bound_property(self):
        """Test that the lipschitz_bound property returns a positive scalar."""
        layer = LipschitzLinear(4, 2)
        bound = layer.lipschitz_bound
        
        assert isinstance(bound, torch.Tensor)
        assert bound.numel() == 1  # Should be scalar
        assert bound.item() > 0  # Should be positive
        assert bound.requires_grad  # Should be differentiable

    def test_lipschitz_regularization_reduces_bound(self):
        """Test that Lipschitz regularization actually reduces the bound."""
        # Create a simple model with one LipschitzLinear layer
        model = LipschitzLinear(4, 2)
        
        # Create some dummy data
        batch_size = 10
        input_data = torch.randn(batch_size, 4)
        target_data = torch.randn(batch_size, 2)
        
        # Get initial Lipschitz bound
        initial_bound = model.lipschitz_bound.item()
        
        # Setup optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        # Training loop with Lipschitz regularization
        lambda_lipschitz = 1000.0  # Strong regularization
        
        for epoch in range(50):
            optimizer.zero_grad()
            
            # Forward pass
            output = model(input_data)
            
            # Compute loss with Lipschitz regularization
            reconstruction_loss = nn.MSELoss()(output, target_data)
            
            # Manual Lipschitz regularization (mimic what happens in the full model)
            log_lipschitz_bound = torch.log10(model.lipschitz_bound)
            lipschitz_regularization = log_lipschitz_bound
            
            # Total loss (adding the regularization term)
            total_loss = reconstruction_loss + lambda_lipschitz * lipschitz_regularization
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
        
        final_bound = model.lipschitz_bound.item()
        
        # Assert that the bound decreased
        assert final_bound < initial_bound, (
            f"Lipschitz bound should decrease with regularization. "
            f"Initial: {initial_bound:.4f}, Final: {final_bound:.4f}"
        )
        
        # Assert that the change is significant (more than 10% reduction)
        relative_change = (initial_bound - final_bound) / initial_bound
        assert relative_change > 0.1, (
            f"Lipschitz bound should decrease significantly. "
            f"Relative change: {relative_change:.2%}"
        )


class TestAEWithLipschitz:
    """Test the autoencoder with Lipschitz constraints."""
    
    def test_ae_log_lipschitz_bounds(self):
        """Test that log_lipschitz_bounds property works correctly."""
        # Create a small autoencoder with Lipschitz layers
        model = AE(
            input_dim=(3, 4),  # 2D input  
            hidden_dims=[8, 4],
            latent_dim=2,
            use_lipschitz_layers=True
        )
        
        # Initialize normalization with dummy data
        dummy_data = torch.randn(10, 3, 4)
        model.initialize_normalization(dummy_data)
        
        # Test that log_lipschitz_bounds returns tuple of scalars
        encoder_bound, decoder_bound = model.log_lipschitz_bounds
        
        assert isinstance(encoder_bound, torch.Tensor)
        assert isinstance(decoder_bound, torch.Tensor) 
        assert encoder_bound.numel() == 1
        assert decoder_bound.numel() == 1
        assert encoder_bound.requires_grad
        assert decoder_bound.requires_grad

    def test_ae_without_lipschitz_raises_error(self):
        """Test that log_lipschitz_bounds raises error when use_lipschitz_layers=False."""
        model = AE(
            input_dim=12,
            hidden_dims=[8, 4],
            latent_dim=2,
            use_lipschitz_layers=False
        )
        
        with pytest.raises(RuntimeError, match="Model does not contain any LipschitzLinear layers"):
            _ = model.log_lipschitz_bounds


class TestLipschitzLossFunction:
    """Test the loss function with Lipschitz regularization."""
    
    def test_loss_function_with_lipschitz(self):
        """Test that the loss function correctly incorporates Lipschitz regularization."""
        batch_size = 5
        input_dim = 8
        latent_dim = 3
        
        # Create dummy data
        input_vector = torch.randn(batch_size, input_dim)
        reconstructed_vector = torch.randn(batch_size, input_dim)
        latent_vector = torch.randn(batch_size, latent_dim)
        
        # Create dummy log Lipschitz bounds
        log_lipschitz_bounds = (
            torch.tensor(0.5, requires_grad=True),  # log10 of encoder bound
            torch.tensor(0.3, requires_grad=True)   # log10 of decoder bound
        )
        
        # Test with Lipschitz regularization
        lambda_lipschitz = 1.
        
        total_loss = loss_function(
            input_vector=input_vector,
            reconstructed_vector=reconstructed_vector,
            latent_vector=latent_vector,
            log_lipschitz_bounds=log_lipschitz_bounds,
            lambda_lipschitz_encoder=lambda_lipschitz,
            lambda_lipschitz_decoder=lambda_lipschitz
        )
        
        # Test that loss requires gradients
        assert total_loss.requires_grad
        
        # Test that backprop works
        total_loss.backward()
        assert log_lipschitz_bounds[0].grad is not None
        assert log_lipschitz_bounds[1].grad is not None
        
        # Test with return_components=True
        loss_components = loss_function(
            input_vector=input_vector,
            reconstructed_vector=reconstructed_vector,
            latent_vector=latent_vector,
            log_lipschitz_bounds=log_lipschitz_bounds,
            lambda_lipschitz_encoder=lambda_lipschitz,
            lambda_lipschitz_decoder=lambda_lipschitz,
            return_components=True
        )
        
        assert len(loss_components) == 6  # total, recon, norm, repulsion, lip_enc, lip_dec
        total_loss, recon_loss, norm_loss, repul_loss, lip_enc, lip_dec = loss_components
        
        # Check that Lipschitz components match the bounds
        assert torch.allclose(lip_enc, log_lipschitz_bounds[0])
        assert torch.allclose(lip_dec, log_lipschitz_bounds[1])

    def test_loss_function_raises_error_without_bounds(self):
        """Test that loss function raises error when lambda_lipschitz > 0 but no bounds provided."""
        input_vector = torch.randn(5, 8)
        reconstructed_vector = torch.randn(5, 8)
        latent_vector = torch.randn(5, 3)
        
        with pytest.raises(ValueError):
            loss_function(
                input_vector=input_vector,
                reconstructed_vector=reconstructed_vector,
                latent_vector=latent_vector,
                log_lipschitz_bounds=None,
                lambda_lipschitz_encoder=1.0,
                lambda_lipschitz_decoder=1.0
            )
