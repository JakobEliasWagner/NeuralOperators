import warnings
from functools import wraps


def run_once(method):
    """
    Decorator to ensure a method is only run once per object instance.
    Issues a warning if attempted to run more than once.
    """

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, "_method_ran"):
            self._method_ran = set()

        if method.__name__ in self._method_ran:
            warnings.warn(f"Method '{method.__name__}' has already been run for this instance.")
            return

        self._method_ran.add(method.__name__)
        return method(self, *args, **kwargs)

    return wrapper
