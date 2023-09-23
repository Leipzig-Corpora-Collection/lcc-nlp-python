import functools
import weakref
from typing import Iterable
from typing import TypeVar

# ---------------------------------------------------------------------------


__all__ = ["memoized_method", "tqdm"]


# ---------------------------------------------------------------------------


# from: https://stackoverflow.com/a/33672499/9360161
def memoized_method(*lru_args, **lru_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def wrapped_func(self, *args, **kwargs):
            # We're storing the wrapped method inside the instance. If we had
            # a strong reference to self the instance would never die.
            self_weak = weakref.ref(self)

            @functools.wraps(func)
            @functools.lru_cache(*lru_args, **lru_kwargs)
            def cached_method(*args, **kwargs):
                return func(self_weak(), *args, **kwargs)

            object.__setattr__(self, func.__name__, cached_method)
            return cached_method(*args, **kwargs)

        return wrapped_func

    return decorator


# ---------------------------------------------------------------------------


try:
    from tqdm import tqdm
except ImportError:
    T = TypeVar("T", bound=Iterable)

    def tqdm(iterable: T, *args, **kwargs) -> T:  # type: ignore[no-redef]
        return iterable


# ---------------------------------------------------------------------------
