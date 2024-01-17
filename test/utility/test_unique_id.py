from src.utility import get_unique_id


def test_can_generate_id():
    u_id = get_unique_id()
    assert isinstance(u_id, str)
    assert len(u_id) > 0


def test_id_unique():
    test_size = 10000
    ids = set([get_unique_id() for _ in range(test_size)])

    assert len(ids) == test_size
