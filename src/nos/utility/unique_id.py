import random
import warnings
from datetime import datetime
from uuid import UUID, uuid4


class UniqueId:
    def __init__(self, time_stamp: datetime.time = datetime.now(), seed: int = None):
        """Unique ID in the format YYYYMMDDHHMMSS-uuid4.

        Args:
            time_stamp: Time used for the first part of the ID.
            seed: Seed for generating the uuid4 part of the ID.
        """
        # time part
        self.time_stamp = time_stamp

        # uuid part
        if seed is not None:
            warnings.warn("Setting the seed for generating unique ID may compromise thread safety!")
            rng = random.Random()
            rng.seed(seed)  # defaults to system time when None -> thread safety compromised
            self.uuid = UUID(int=rng.getrandbits(128), version=4)
        else:
            self.uuid = uuid4()

    def __str__(self):
        return self.time_stamp.strftime("%Y%m%d%H%M%S-") + str(self.uuid)
