import re

from nos.utils import (
    UniqueId,
)


def test_can_initialize():
    uid = UniqueId()
    assert isinstance(uid, UniqueId)


def test_get_str():
    uid = UniqueId()
    str_uid = str(uid)
    assert isinstance(str_uid, str)


def test_str_structure():
    uid = UniqueId()
    str_uid = str(uid)
    pattern = r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-4[0-9a-fA-F]{3}-[89ABab][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$"  # noqa: E501
    assert re.match(pattern, str_uid)
