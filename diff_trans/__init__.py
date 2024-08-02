import os


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_DEBUG_NANS"] = "true"
os.environ["JAX_ENABLE_X64"] = "true"
# JAX_ENABLE_X64