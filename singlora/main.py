import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import threading


# -----------------------------------------------------------------------------
# 1. SingLoRA Layer Implementation
# -----------------------------------------------------------------------------
class SingLoRALayer(nn.Module):
    """
    Implements the SingLoRA layer as described in the paper.
    This layer wraps a frozen pre-trained layer (e.g., nn.Linear) and
    adds a low-rank update using a single matrix 'A'.

    The weight update is calculated as W = W_0 + alpha/r * u(t) * A @ A.T
    """

    def __init__(
        self,
        original_layer: nn.Module,
        rank: int,
        alpha: float,
        ramp_up_steps: int,
    ):
        """
        Args:
            original_layer (nn.Module): The pre-trained layer to be adapted.
                                        Must be a nn.Linear layer.
            rank (int): The rank 'r' of the low-rank adaptation.
            alpha (float): The scaling factor for the adaptation.
            ramp_up_steps (int): The number of steps 'T' for the ramp-up function u(t).
        """
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.ramp_up_steps = ramp_up_steps

        # Freeze the original layer's parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Determine the dimensions for matrix A based on the paper's extension
        # to non-square matrices.
        self.d_out, self.d_in = out_features, in_features
        if self.d_in > self.d_out:
            # If in_features > out_features, swap them for the logic
            self.d_out, self.d_in = self.d_in, self.d_out

        # Create the single low-rank matrix 'A'
        self.A = nn.Parameter(torch.Tensor(self.d_out, self.rank))

        # Initialize 'A' using Kaiming uniform distribution, as suggested
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # Register a buffer for the training step counter 't'
        self.register_buffer(
            "training_step", torch.tensor(0, dtype=torch.float32)
        )

    def _get_update_weight(self) -> torch.Tensor:
        """
        Calculates the low-rank weight update matrix.
        Handles the ramp-up function u(t) and non-square matrices.
        """
        # Ramp-up function u(t) = min(t/T, 1)
        ramp_up_factor = torch.min(
            self.training_step / self.ramp_up_steps, torch.tensor(1.0)
        ).item()

        # Scaling factor from the paper
        scale = self.alpha / self.rank

        # Calculate A @ A.T
        aa_t = self.A @ self.A.T

        # For non-square matrices, we need to truncate or pad the update
        # as described in Section 4.4 of the paper.
        if (
            self.original_layer.in_features
            > self.original_layer.out_features
        ):
            # A_star is a truncation of A
            update = aa_t[
                : self.original_layer.out_features,
                : self.original_layer.in_features,
            ]
        else:
            # A_star is a truncation of A
            A_star = self.A[: self.d_in, :]
            update = A_star @ self.A.T

        return ramp_up_factor * scale * update

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the SingLoRA layer.
        """
        # Increment the training step counter for the ramp-up function
        if (
            self.training.is_set()
            if isinstance(self.training, threading.Event)
            else self.training
        ):
            self.training_step += 1

        # Get the low-rank update
        update_weight = self._get_update_weight()

        # Get the original frozen weight
        original_weight = self.original_layer.weight

        # Combine the weights: W = W_0 + update
        combined_weight = original_weight + update_weight

        # Perform the linear operation with the adapted weights
        return F.linear(x, combined_weight, self.original_layer.bias)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(rank={self.rank}, alpha={self.alpha}, "
            f"ramp_up_steps={self.ramp_up_steps}, "
            f"original_layer={self.original_layer})"
        )


# -----------------------------------------------------------------------------
# 2. Model Integration Utility
# -----------------------------------------------------------------------------


def apply_singlora_to_model(
    model: nn.Module,
    rank: int,
    alpha: float,
    ramp_up_steps: int,
    target_modules=["query", "key", "value"],
):
    """
    Recursively replaces target linear layers in a model with SingLoRALayer.

    Args:
        model (nn.Module): The model to be modified.
        rank (int): The rank for SingLoRA.
        alpha (float): The alpha scaling for SingLoRA.
        ramp_up_steps (int): The T parameter for the ramp-up function.
        target_modules (list[str]): A list of substrings to identify target layers
                                    by name (e.g., 'query', 'key' for attention).
    """
    for name, module in model.named_children():
        if any(
            target in name.lower() for target in target_modules
        ) and isinstance(module, nn.Linear):
            # Replace the identified linear layer with a SingLoRALayer
            setattr(
                model,
                name,
                SingLoRALayer(module, rank, alpha, ramp_up_steps),
            )
            print(f"Replaced '{name}' with SingLoRA layer.")
        else:
            # Recursively apply to child modules
            apply_singlora_to_model(
                module, rank, alpha, ramp_up_steps, target_modules
            )
