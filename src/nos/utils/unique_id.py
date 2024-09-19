from datetime import datetime  # noqa: D100
from uuid import uuid4


class UniqueId:
    """Unique ID."""

    def __init__(self, time_stamp: datetime | None = None) -> None:
        """Unique-ID YYYY_MM_DD_HH_MM_SS-uuid4.

        Args:
            time_stamp: Time used for the first part of the ID.

        """
        # time part
        self.time_stamp = datetime.now() if time_stamp is None else time_stamp  # noqa: DTZ005
        self.uuid = uuid4()

    def __str__(self) -> str:
        """Return UID as string."""
        return self.time_stamp.strftime("%Y_%m_%d_%H_%M_%S-") + str(self.uuid)
