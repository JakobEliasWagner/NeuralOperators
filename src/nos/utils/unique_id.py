from datetime import (
    datetime,
)
from uuid import (
    uuid4,
)


class UniqueId:
    """Unique ID."""

    def __init__(self, time_stamp: datetime.time = datetime.now()):
        """Unique ID in the format YYYY_MM_DD_HH_MM_SS-uuid4.

        Args:
            time_stamp: Time used for the first part of the ID.
        """
        # time part
        self.time_stamp = time_stamp
        self.uuid = uuid4()

    def __str__(self):
        return self.time_stamp.strftime("%Y_%m_%d_%H_%M_%S-") + str(self.uuid)
