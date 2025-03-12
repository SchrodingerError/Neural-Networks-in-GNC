import torch
from typing import Union, List

def one_ninth_loss(theta: torch.Tensor, desired_theta: torch.Tensor) -> torch.Tensor:
    """
    Computes the error loss raised to the power of 1/9: |theta - desired_theta|^(1/9)
    """
    return torch.abs(theta - desired_theta) ** (1/9)

def one_eighth_loss(theta: torch.Tensor, desired_theta: torch.Tensor) -> torch.Tensor:
    """
    Computes the error loss raised to the power of 1/8: |theta - desired_theta|^(1/8)
    """
    return torch.abs(theta - desired_theta) ** (1/8)

def one_fourth_loss(theta: torch.Tensor, desired_theta: torch.Tensor) -> torch.Tensor:
    """
    Computes the error loss raised to the power of 1/4: |theta - desired_theta|^(1/4)
    """
    return torch.abs(theta - desired_theta) ** (1/4)

def one_third_loss(theta: torch.Tensor, desired_theta: torch.Tensor) -> torch.Tensor:
    """
    Computes the error loss raised to the power of 1/3: |theta - desired_theta|^(1/3)
    """
    return torch.abs(theta - desired_theta) ** (1/3)

def one_half_loss(theta: torch.Tensor, desired_theta: torch.Tensor) -> torch.Tensor:
    """
    Computes the error loss raised to the power of 1/2: |theta - desired_theta|^(1/2)
    """
    return torch.abs(theta - desired_theta) ** (1/2)

def abs_loss(theta: torch.Tensor, desired_theta: torch.Tensor) -> torch.Tensor:
    """
    Computes the absolute error loss: |theta - desired_theta| (exponent 1)
    """
    return torch.abs(theta - desired_theta)

def square_loss(theta: torch.Tensor, desired_theta: torch.Tensor) -> torch.Tensor:
    """
    Computes the squared error loss: |theta - desired_theta|^2
    """
    return torch.abs(theta - desired_theta) ** 2

def cube_loss(theta: torch.Tensor, desired_theta: torch.Tensor) -> torch.Tensor:
    """
    Computes the cubed error loss: |theta - desired_theta|^3
    """
    return torch.abs(theta - desired_theta) ** 3

def fourth_loss(theta: torch.Tensor, desired_theta: torch.Tensor) -> torch.Tensor:
    """
    Computes the error loss raised to the power of 4: |theta - desired_theta|^4
    """
    return torch.abs(theta - desired_theta) ** 4

def eight_loss(theta: torch.Tensor, desired_theta: torch.Tensor) -> torch.Tensor:
    """
    Computes the error loss raised to the power of 8: |theta - desired_theta|^8
    """
    return torch.abs(theta - desired_theta) ** 8

def nine_loss(theta: torch.Tensor, desired_theta: torch.Tensor) -> torch.Tensor:
    """
    Computes the error loss raised to the power of 9: |theta - desired_theta|^9
    """
    return torch.abs(theta - desired_theta) ** 9

# Dictionary to store function references along with their corresponding exponent.
base_loss_functions = {
    'one_ninth': (1/9, one_ninth_loss),
    'one_eighth': (1/8, one_eighth_loss),
    'one_fourth': (1/4, one_fourth_loss),
    'one_third': (1/3, one_third_loss),
    'one_half': (1/2, one_half_loss),
    'abs': (1, abs_loss),
    'square': (2, square_loss),
    'cube': (3, cube_loss),
    'four': (4, fourth_loss),
    'eight': (8, eight_loss),
    'nine': (9, nine_loss)
}
