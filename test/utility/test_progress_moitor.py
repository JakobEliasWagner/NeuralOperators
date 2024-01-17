import multiprocessing as mp

from src.utility import ProgressMonitor


def dummy(i):
    return i


def dummy_q(args):
    i, q = args
    q.put(i)
    return dummy(i)


def test_sequential():
    test_len = 1000
    for i in range(test_len):
        dummy(i)
        ProgressMonitor.print_progress_bar(i, test_len)
    assert True


def test_threads_available():
    """Tests whether the testing run has access to multiple cores"""
    num_threads = min([2, mp.cpu_count()])
    assert num_threads > 1


def test_multiple_threads():
    test_len = 1000
    num_threads = min([2, mp.cpu_count()])
    pool = mp.Pool(processes=num_threads)
    manager = mp.Manager()
    queue = manager.Queue()

    args = [(i, queue) for i in range(test_len)]

    result = pool.map_async(dummy, args)

    ProgressMonitor.monitor_pool(result, queue, test_len)

    # monitor has returned -> no fatal exception
    assert True
