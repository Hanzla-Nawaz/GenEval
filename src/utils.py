# utils.py in phi package

import inspect

def get_method_sig(method):
    """Get the method signature."""
    signature = inspect.signature(method)
    return str(signature)
