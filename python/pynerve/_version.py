from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pynerve")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
