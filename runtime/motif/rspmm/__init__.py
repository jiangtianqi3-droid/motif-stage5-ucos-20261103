def generalized_rspmm(*args, **kwargs):
    from .rspmm import generalized_rspmm as implementation
    return implementation(*args, **kwargs)
