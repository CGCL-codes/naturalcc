from typing import Iterable, Any, Tuple, Optional


def find_innermost_interval(
    intervals: Iterable[Tuple[int, int, Any]], index: int
) -> Optional[Tuple[int, int, Any]]:
    """
    Finds the innermost interval that contains the specified index.

    Args:
    intervals: A list of intervals, where each interval is a tuple of two integers and an object (start, end, Any).
    index: An integer index that should be contained within the intervals.

    Returns:
    The innermost interval as a tuple (start, end, Any) that contains the index, or None if no such interval exists.
    """

    # Initialize a variable to save the innermost interval, starting with None (no interval)
    inner_most_interval = None

    # Loop through all the intervals to find ones that contain the index
    for interval in intervals:
        start, end, _ = interval
        # Check if the interval contains the index
        if start <= index <= end:
            # If there's no innermost interval yet, or if the current interval is smaller, update the innermost interval
            if inner_most_interval is None or (end - start) < (
                inner_most_interval[1] - inner_most_interval[0]
            ):
                inner_most_interval = interval

    return inner_most_interval
