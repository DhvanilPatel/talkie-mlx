"""Unofficial MLX support for the Talkie-1930 model family."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("talkie-mlx")
except PackageNotFoundError:  # pragma: no cover - editable tree before install
    __version__ = "0.0.0"

__all__ = ["__version__"]
