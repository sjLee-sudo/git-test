from multiprocessing import Pool, cpu_count, Process


def pool_multiprocess(workers_num, target_function,iterable_list):
    with Pool(workers_num) as pool:
        pool.map(target_function, iterable_list)
        pool.close()
        pool.join()