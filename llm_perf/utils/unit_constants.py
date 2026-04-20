"""Unit conversion constants and functions.

This module provides constants and helper functions for unit conversions
commonly used in LLM performance calculations.
"""

# Byte unit conversions
BYTES_PER_GB: float = 1e9
BYTES_PER_MB: float = 1e6
BYTES_PER_KB: float = 1e3

# FLOPs unit conversions
FLOPS_PER_GFLOPS: float = 1e9
FLOPS_PER_TFLOPS: float = 1e12
FLOPS_PER_PFLOPS: float = 1e15

# Parameter unit conversions
PARAMS_PER_BILLION: float = 1e9
PARAMS_PER_MILLION: float = 1e6


def bytes_to_gb(bytes_value: float) -> float:
    """Convert bytes to gigabytes.

    Args:
        bytes_value: Value in bytes.

    Returns:
        Value in gigabytes.
    """
    return bytes_value / BYTES_PER_GB


def bytes_to_mb(bytes_value: float) -> float:
    """Convert bytes to megabytes.

    Args:
        bytes_value: Value in bytes.

    Returns:
        Value in megabytes.
    """
    return bytes_value / BYTES_PER_MB


def flops_to_gflops(flops_value: float) -> float:
    """Convert FLOPs to GFLOPs.

    Args:
        flops_value: Value in FLOPs.

    Returns:
        Value in GFLOPs.
    """
    return flops_value / FLOPS_PER_GFLOPS


def flops_to_tflops(flops_value: float) -> float:
    """Convert FLOPs to TFLOPs.

    Args:
        flops_value: Value in FLOPs.

    Returns:
        Value in TFLOPs.
    """
    return flops_value / FLOPS_PER_TFLOPS


def params_to_billion(params_value: float) -> float:
    """Convert parameters count to billions.

    Args:
        params_value: Number of parameters.

    Returns:
        Number of parameters in billions.
    """
    return params_value / PARAMS_PER_BILLION


def params_to_million(params_value: float) -> float:
    """Convert parameters count to millions.

    Args:
        params_value: Number of parameters.

    Returns:
        Number of parameters in millions.
    """
    return params_value / PARAMS_PER_MILLION
