from datetime import datetime
from uuid import uuid4


def get_unique_id() -> str:
    """Get a unique id.

    Returns: unique id.

    """
    return datetime.now().strftime("%Y%m%d%H%M%S-") + str(uuid4())
