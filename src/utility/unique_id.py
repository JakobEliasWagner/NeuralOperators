from datetime import datetime
from uuid import uuid4


def get_unique_id() -> str:
    """Returns a unique id with time and a uuid4

    :return:
    """
    return datetime.now().strftime("%Y%m%d%H%M%S-") + str(uuid4())
