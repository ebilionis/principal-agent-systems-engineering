"""
This files contains some functions.
"""
__all__ = ['split']

def split(container, count):
    """
    split the jobs for parallel run
    """
    return [container[_i::count] for _i in range(count)]
