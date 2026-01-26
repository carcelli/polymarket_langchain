import collections
from typing import Any, Mapping


class StrKeyDict(collections.UserDict):
    """
    A dictionary that coerces all keys to strings on access and insertion.

    Use Cases:
    - Normalizing Market IDs (which come as ints from CLOB and strings from Gamma).
    - Caching tool results where the input might vary in type.
    """

    def __missing__(self, key: Any) -> Any:
        """
        Called when a key is missing.
        If the key wasn't a string, convert it and try again.
        """
        if isinstance(key, str):
            raise KeyError(key)
        return self[str(key)]

    def __contains__(self, key: Any) -> bool:
        """Check for existence using the stringified version."""
        return str(key) in self.data

    def __setitem__(self, key: Any, item: Any) -> None:
        """Always store keys as strings to maintain consistency."""
        self.data[str(key)] = item

    def get(self, key: Any, default: Any = None) -> Any:
        """Safe retrieval that respects the string coercion."""
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, other=(), **kwds):
        """
        Custom update to ensure batch operations also respect string coercion.
        """
        if isinstance(other, Mapping):
            for k, v in other.items():
                self[k] = v
        elif hasattr(other, "keys"):  # Mapping-like
            for k in other.keys():
                self[k] = other[k]
        else:  # Iterable of pairs
            for k, v in other:
                self[k] = v
        for k, v in kwds.items():
            self[k] = v
