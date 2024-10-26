import sys

def export(fn):
    module = sys.modules[fn.__module__]
    if hasattr(module, '__all__'):
        module.__all__.append(fn.__name__)
    else:
        module.__all__ = [fn.__name__]
    return fn
