from multiprocessing import Pool
from tqdm import tqdm

def map_reduce(data, map_fct, initial_value, reduce_fct, pool_size=None, chunksize=32, desc=None, no_parallel=False):
    """
    Starts a pool of multiple processes which each apply map_fct to a subset of 
    items in the data iterable.

    The result of each map operation is reduced using the recurrence 
    relation existing = reduce_fct(existing, result).

    This operation may not be order-preserving.

    Parameters
    ----------
    no_parallel: bool
        Specify this to run in a non-parallel debug mode, if better stacktraces are required.
    """
    # simple non-parallel debug version
    if no_parallel:
        v = initial_value
        for item in tqdm(data, desc=desc):
            res = map_fct(item)
            v = reduce_fct(v, res)
        return v

    pool = Pool(pool_size)
    reduce_result = initial_value
    try:
        for map_res in tqdm(pool.imap_unordered(map_fct, data, chunksize=chunksize), desc=desc, total=len(data)):
            reduce_result = reduce_fct(reduce_result, map_res)
    except KeyboardInterrupt:
        pool.terminate()
    pool.close()
    pool.join()
    return reduce_result