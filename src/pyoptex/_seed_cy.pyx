from libc.stdlib cimport srand

def set_seed_cy(n):
    srand(n)
