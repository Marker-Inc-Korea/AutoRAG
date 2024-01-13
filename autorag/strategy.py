import time


def measure_time(func, *args, **kwargs):
    """
    Method for measuring execution time of the function.
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time


