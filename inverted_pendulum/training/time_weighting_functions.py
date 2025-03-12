import torch
from typing import Union, List

def constant(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    return torch.ones_like(t_span)

def linear(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return min_val + ((1 - min_val) / (t_max)**1) * t_span**1

def quadratic(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return min_val + ((1 - min_val) / (t_max)**2) * t_span**2

def cubic(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return min_val + ((1 - min_val) / (t_max)**3) * t_span**3

def square_root(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return min_val + ((1 - min_val) / (t_max)**(1/2)) * t_span**(1/2)

def cubic_root(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return min_val + ((1 - min_val) / (t_max)**(1/3)) * t_span**(1/3)

def inverse(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return (((1/min_val)**(1/1) - 1) * 1/t_max * t_span + 1)**-1

def inverse_squared(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return (((1/min_val)**(1/2) - 1) * 1/t_max * t_span + 1)**-2

def inverse_cubed(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return (((1/min_val)**(1/3) - 1) * 1/t_max * t_span + 1)**-3

def linear_mirrored(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return min_val + ((1 - min_val) / (t_max)**1) * (-t_span + t_max)**1

def quadratic_mirrored(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return min_val + ((1 - min_val) / (t_max)**2) * (-t_span + t_max)**2

def cubic_mirrored(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return min_val + ((1 - min_val) / (t_max)**3) * (-t_span + t_max)**3

def square_root_mirrored(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return min_val + ((1 - min_val) / (t_max)**(1/2)) * (-t_span + t_max)**(1/2)

def cubic_root_mirrored(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return min_val + ((1 - min_val) / (t_max)**(1/3)) * (-t_span + t_max)**(1/3)

def inverse_mirrored(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return (((1/min_val)**(1/1) - 1) * 1/t_max * (-t_span + t_max) + 1)**-1

def inverse_squared_mirrored(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return (((1/min_val)**(1/2) - 1) * 1/t_max * (-t_span + t_max) + 1)**-2

def inverse_cubed_mirrored(t_span: Union[torch.Tensor, List[float]], t_max: float = None, min_val: float = 0.01) -> torch.Tensor:
    t_span = t_span.clone().detach() if isinstance(t_span, torch.Tensor) else torch.tensor(t_span)
    t_max = t_max if t_max is not None else t_span[-1]
    return (((1/min_val)**(1/3) - 1) * 1/t_max * (-t_span + t_max) + 1)**-3

# Dictionary to store function references
weight_functions = {
    'constant': constant,
    'linear': linear,
    'quadratic': quadratic,
    'cubic': cubic,
    'square_root': square_root,
    'cubic_root': cubic_root,
    'inverse': inverse,
    'inverse_squared': inverse_squared,
    'inverse_cubed': inverse_cubed,
    'linear_mirrored': linear_mirrored,
    'quadratic_mirrored': quadratic_mirrored,
    'cubic_mirrored': cubic_mirrored,
    'square_root_mirrored': square_root_mirrored,
    'cubic_root_mirrored': cubic_root_mirrored,
    'inverse_mirrored': inverse_mirrored,
    'inverse_squared_mirrored': inverse_squared_mirrored,
    'inverse_cubed_mirrored': inverse_cubed_mirrored,
}