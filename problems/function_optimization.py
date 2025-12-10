"""
Benchmark Optimization Functions
"""

import numpy as np
from typing import Callable, Tuple, Optional


def sphere_function(x: np.ndarray) -> float:
    """
    Sphere function (unimodal, convex).
    
    Global minimum: f(0, 0, ..., 0) = 0
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector
    
    Returns:
    --------
    float : Function value
    """
    return np.sum(x ** 2)


def rastrigin_function(x: np.ndarray) -> float:
    """
    Rastrigin function (highly multimodal).
    
    Global minimum: f(0, 0, ..., 0) = 0
    Typical search domain: [-5.12, 5.12]^n
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector
    
    Returns:
    --------
    float : Function value
    """
    n = len(x)
    A = 10
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))


def ackley_function(x: np.ndarray) -> float:
    """
    Ackley function (multimodal, many local minima).
    
    Global minimum: f(0, 0, ..., 0) = 0
    Typical search domain: [-32, 32]^n
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector
    
    Returns:
    --------
    float : Function value
    """
    n = len(x)
    a = 20
    b = 0.2
    c = 2 * np.pi
    
    sum1 = np.sum(x ** 2)
    sum2 = np.sum(np.cos(c * x))
    
    term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
    term2 = -np.exp(sum2 / n)
    
    return term1 + term2 + a + np.e


def rosenbrock_function(x: np.ndarray) -> float:
    """
    Rosenbrock function (valley-shaped, unimodal but hard to optimize).
    
    Global minimum: f(1, 1, ..., 1) = 0
    Typical search domain: [-5, 10]^n
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector
    
    Returns:
    --------
    float : Function value
    """
    return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)


def griewank_function(x: np.ndarray) -> float:
    """
    Griewank function (multimodal with many local minima).
    
    Global minimum: f(0, 0, ..., 0) = 0
    Typical search domain: [-600, 600]^n
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector
    
    Returns:
    --------
    float : Function value
    """
    sum_term = np.sum(x ** 2) / 4000
    prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return 1 + sum_term - prod_term


def schwefel_function(x: np.ndarray) -> float:
    """
    Schwefel function (highly non-symmetrical, deceptive).
    
    Global minimum: f(420.9687, 420.9687, ..., 420.9687) ≈ 0
    Typical search domain: [-500, 500]^n
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector
    
    Returns:
    --------
    float : Function value
    """
    n = len(x)
    return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def shifted_sphere_function(x: np.ndarray, shift: np.ndarray = None) -> float:
    """
    Shifted Sphere function (non-symmetrical, shifted from origin).
    
    Global minimum: f(shift) = 0
    Typical search domain: [-5.12, 5.12]^n
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector
    shift : np.ndarray
        Shift vector (default: [2.0, -1.5])
    
    Returns:
    --------
    float : Function value
    """
    if shift is None:
        shift = np.array([2.0, -1.5])
    shifted_x = x - shift
    return np.sum(shifted_x ** 2)


def rotated_ellipsoid_function(x: np.ndarray) -> float:
    """
    Rotated Ellipsoid function (non-symmetrical, rotated).
    
    Global minimum: f(0, 0) = 0
    Typical search domain: [-5.12, 5.12]^n
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector
    
    Returns:
    --------
    float : Function value
    """
    if len(x) != 2:
        raise ValueError("Rotated ellipsoid currently supports only 2D")
    
    # Rotation angle (45 degrees)
    theta = np.pi / 4
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Rotation matrix
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    rotated_x = R @ x
    
    # Ellipsoid with different scaling
    return 5 * rotated_x[0]**2 + 0.5 * rotated_x[1]**2


def beale_function(x: np.ndarray) -> float:
    """
    Beale function (non-symmetrical, valley-shaped).
    
    Global minimum: f(3, 0.5) = 0
    Typical search domain: [-4.5, 4.5]^2
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector (2D)
    
    Returns:
    --------
    float : Function value
    """
    if len(x) != 2:
        raise ValueError("Beale function requires 2D input")
    
    x1, x2 = x[0], x[1]
    term1 = (1.5 - x1 + x1 * x2)**2
    term2 = (2.25 - x1 + x1 * x2**2)**2
    term3 = (2.625 - x1 + x1 * x2**3)**2
    return term1 + term2 + term3


def goldstein_price_function(x: np.ndarray) -> float:
    """
    Goldstein-Price function (non-symmetrical, multimodal).
    
    Global minimum: f(0, -1) = 3
    Typical search domain: [-2, 2]^2
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector (2D)
    
    Returns:
    --------
    float : Function value
    """
    if len(x) != 2:
        raise ValueError("Goldstein-Price function requires 2D input")
    
    x1, x2 = x[0], x[1]
    term1 = 1 + (x1 + x2 + 1)**2 * (19 - 14*x1 + 3*x1**2 - 14*x2 + 6*x1*x2 + 3*x2**2)
    term2 = 30 + (2*x1 - 3*x2)**2 * (18 - 32*x1 + 12*x1**2 + 48*x2 - 36*x1*x2 + 27*x2**2)
    return term1 * term2


def three_hump_camel_function(x: np.ndarray) -> float:
    """
    Three-Hump Camel function (non-symmetrical, three local minima).
    
    Global minimum: f(0, 0) = 0
    Typical search domain: [-5, 5]^2
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector (2D)
    
    Returns:
    --------
    float : Function value
    """
    if len(x) != 2:
        raise ValueError("Three-Hump Camel function requires 2D input")
    
    x1, x2 = x[0], x[1]
    return 2*x1**2 - 1.05*x1**4 + x1**6/6 + x1*x2 + x2**2


def easom_function(x: np.ndarray) -> float:
    """
    Easom function (non-symmetrical, very flat with sharp global minimum).
    
    Global minimum: f(π, π) = -1
    Typical search domain: [-100, 100]^2
    
    Parameters:
    -----------
    x : np.ndarray
        Input vector (2D)
    
    Returns:
    --------
    float : Function value
    """
    if len(x) != 2:
        raise ValueError("Easom function requires 2D input")
    
    x1, x2 = x[0], x[1]
    return -np.cos(x1) * np.cos(x2) * np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))


def get_function_gradient(func_name: str) -> Optional[Callable]:
    """
    Get gradient function for a given benchmark function.
    
    Parameters:
    -----------
    func_name : str
        Name of the function
    
    Returns:
    --------
    Optional[Callable] : Gradient function, or None if not available
    """
    if func_name == 'sphere':
        def gradient(x):
            return 2 * x
        return gradient
    
    elif func_name == 'rastrigin':
        def gradient(x):
            A = 10
            return 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)
        return gradient
    
    elif func_name == 'ackley':
        def gradient(x):
            n = len(x)
            a = 20
            b = 0.2
            c = 2 * np.pi
            
            sum1 = np.sum(x ** 2)
            sum2 = np.sum(np.cos(c * x))
            
            term1 = (a * b / (n * np.sqrt(sum1 / n))) * np.exp(-b * np.sqrt(sum1 / n)) * x
            term2 = (c / n) * np.exp(sum2 / n) * np.sin(c * x)
            
            return term1 + term2
        return gradient
    
    elif func_name == 'rosenbrock':
        def gradient(x):
            grad = np.zeros_like(x)
            grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
            for i in range(1, len(x) - 1):
                grad[i] = 200 * (x[i] - x[i-1]**2) - 400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
            grad[-1] = 200 * (x[-1] - x[-2]**2)
            return grad
        return gradient
    
    elif func_name == 'griewank':
        def gradient(x):
            n = len(x)
            sum_term = x / 2000
            
            # Product term gradient (complex)
            prod = np.prod(np.cos(x / np.sqrt(np.arange(1, n + 1))))
            prod_grad = np.zeros_like(x)
            for i in range(n):
                term = -np.sin(x[i] / np.sqrt(i + 1)) / np.sqrt(i + 1)
                for j in range(n):
                    if j != i:
                        term *= np.cos(x[j] / np.sqrt(j + 1))
                prod_grad[i] = term * prod
            
            return sum_term - prod_grad
        return gradient
    
    elif func_name == 'shifted_sphere':
        def gradient(x):
            shift = np.array([2.0, -1.5])
            return 2 * (x - shift)
        return gradient
    
    elif func_name == 'rotated_ellipsoid':
        def gradient(x):
            if len(x) != 2:
                raise ValueError("Rotated ellipsoid currently supports only 2D")
            theta = np.pi / 4
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
            rotated_x = R @ x
            grad_rotated = np.array([10 * rotated_x[0], rotated_x[1]])
            return R.T @ grad_rotated
        return gradient
    
    elif func_name == 'beale':
        def gradient(x):
            if len(x) != 2:
                raise ValueError("Beale function requires 2D input")
            x1, x2 = x[0], x[1]
            grad_x1 = 2*(1.5 - x1 + x1*x2)*(-1 + x2) + \
                      2*(2.25 - x1 + x1*x2**2)*(-1 + x2**2) + \
                      2*(2.625 - x1 + x1*x2**3)*(-1 + x2**3)
            grad_x2 = 2*(1.5 - x1 + x1*x2)*x1 + \
                      2*(2.25 - x1 + x1*x2**2)*2*x1*x2 + \
                      2*(2.625 - x1 + x1*x2**3)*3*x1*x2**2
            return np.array([grad_x1, grad_x2])
        return gradient
    
    else:
        # For functions without explicit gradients, return None
        # This allows gradient descent to skip these functions
        return None


# Function metadata
FUNCTION_INFO = {
    'sphere': {
        'name': 'Sphere',
        'function': sphere_function,
        'bounds': np.array([[-5.12, 5.12], [-5.12, 5.12]]),
        'global_min': 0.0,
        'global_min_pos': np.array([0.0, 0.0]),
        'description': 'Unimodal, convex function'
    },
    'rastrigin': {
        'name': 'Rastrigin',
        'function': rastrigin_function,
        'bounds': np.array([[-5.12, 5.12], [-5.12, 5.12]]),
        'global_min': 0.0,
        'global_min_pos': np.array([0.0, 0.0]),
        'description': 'Highly multimodal function'
    },
    'ackley': {
        'name': 'Ackley',
        'function': ackley_function,
        'bounds': np.array([[-32, 32], [-32, 32]]),
        'global_min': 0.0,
        'global_min_pos': np.array([0.0, 0.0]),
        'description': 'Multimodal with many local minima'
    },
    'rosenbrock': {
        'name': 'Rosenbrock',
        'function': rosenbrock_function,
        'bounds': np.array([[-5, 10], [-5, 10]]),
        'global_min': 0.0,
        'global_min_pos': np.array([1.0, 1.0]),
        'description': 'Valley-shaped, hard to optimize'
    },
    'griewank': {
        'name': 'Griewank',
        'function': griewank_function,
        'bounds': np.array([[-600, 600], [-600, 600]]),
        'global_min': 0.0,
        'global_min_pos': np.array([0.0, 0.0]),
        'description': 'Multimodal with many local minima'
    },
    'schwefel': {
        'name': 'Schwefel',
        'function': schwefel_function,
        'bounds': np.array([[-500, 500], [-500, 500]]),
        'global_min': 0.0,
        'global_min_pos': np.array([420.9687, 420.9687]),
        'description': 'Highly non-symmetrical, deceptive'
    },
    'shifted_sphere': {
        'name': 'Shifted Sphere',
        'function': lambda x: shifted_sphere_function(x, np.array([2.0, -1.5])),
        'bounds': np.array([[-5.12, 5.12], [-5.12, 5.12]]),
        'global_min': 0.0,
        'global_min_pos': np.array([2.0, -1.5]),
        'description': 'Non-symmetrical, shifted from origin'
    },
    'rotated_ellipsoid': {
        'name': 'Rotated Ellipsoid',
        'function': rotated_ellipsoid_function,
        'bounds': np.array([[-5.12, 5.12], [-5.12, 5.12]]),
        'global_min': 0.0,
        'global_min_pos': np.array([0.0, 0.0]),
        'description': 'Non-symmetrical, rotated ellipsoid'
    },
    'beale': {
        'name': 'Beale',
        'function': beale_function,
        'bounds': np.array([[-4.5, 4.5], [-4.5, 4.5]]),
        'global_min': 0.0,
        'global_min_pos': np.array([3.0, 0.5]),
        'description': 'Non-symmetrical, valley-shaped'
    },
    'goldstein_price': {
        'name': 'Goldstein-Price',
        'function': goldstein_price_function,
        'bounds': np.array([[-2, 2], [-2, 2]]),
        'global_min': 3.0,
        'global_min_pos': np.array([0.0, -1.0]),
        'description': 'Non-symmetrical, multimodal'
    },
    'three_hump_camel': {
        'name': 'Three-Hump Camel',
        'function': three_hump_camel_function,
        'bounds': np.array([[-5, 5], [-5, 5]]),
        'global_min': 0.0,
        'global_min_pos': np.array([0.0, 0.0]),
        'description': 'Non-symmetrical, three local minima'
    },
    'easom': {
        'name': 'Easom',
        'function': easom_function,
        'bounds': np.array([[-100, 100], [-100, 100]]),
        'global_min': -1.0,
        'global_min_pos': np.array([np.pi, np.pi]),
        'description': 'Non-symmetrical, flat with sharp minimum'
    }
}

