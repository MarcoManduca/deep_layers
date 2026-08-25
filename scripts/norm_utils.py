"""Shared helpers for normalization layers across architecture builders."""


def num_groups(filters: int, max_groups: int = 32) -> int:
    """Largest divisor of ``filters`` that is at most ``max_groups``.

    ``GroupNormalization`` requires ``groups`` to divide the channel count
    evenly; a fixed ``32`` (the paper's default) fails on the small filter
    counts unit tests use, so the group count adapts down. ``32`` is used
    unmodified for every real channel size in this project (64-1024,
    see ``fixing.md`` §3).

    Parameters
    ----------
    filters : int
        Number of channels the ``GroupNormalization`` layer will see.
    max_groups : int
        Upper bound on the number of groups.

    Returns
    -------
    int
        A valid ``groups`` value for ``GroupNormalization``.
    """
    groups = min(max_groups, filters)
    while filters % groups != 0:
        groups -= 1
    return groups
