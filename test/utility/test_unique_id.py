import datetime
import multiprocessing as mp
import re
import warnings

from nos.utility import UniqueId


def get_str_id(t) -> str:
    return str(UniqueId(time_stamp=t))


def test_can_generate_id():
    u_id = UniqueId()
    assert isinstance(str(u_id), str)
    assert len(str(u_id)) > 0


def test_id_unique():
    test_size = 10000
    ids = {str(UniqueId()) for _ in range(test_size)}

    assert len(ids) == test_size


def test_id_reproducible():
    seed = 42
    t_stamp = datetime.datetime.now()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        id1 = str(UniqueId(time_stamp=t_stamp, seed=seed))
        id2 = str(UniqueId(time_stamp=t_stamp, seed=seed))

    assert id1 == id2


def test_multiprocessing_safe():
    t_stamp = datetime.datetime.now()
    test_size = 10000

    n_threads = mp.cpu_count()
    assert n_threads > 1

    with mp.Pool(n_threads) as pool:
        data = set(pool.map(get_str_id, [t_stamp for _ in range(test_size)]))

    assert len(data) == test_size


def test_id_format():
    uid = str(UniqueId())
    assert re.match(
        r"\d{4}\d{2}\d{2}\d{2}\d{2}\d{2}-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", uid
    )
