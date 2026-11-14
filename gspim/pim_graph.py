"""Packed publication indices shared by trace construction and scheduling."""

import numpy as np


class ResourceTokens:
    """Allocate the lowest free token ranges, retaining each previous owner."""

    def __init__(self, capacity):
        self.free = [(0, capacity, None, 0)] if capacity else []

    def allocate(self, count):
        ranges, previous = [], []
        consumed = 0
        while count:
            first, last, owner, end = self.free[consumed]
            stop = min(last, first + count)
            ranges.append((first, stop))
            if owner is not None:
                previous.append((owner, end))
            count -= stop - first
            if stop == last:
                consumed += 1
            else:
                self.free[consumed] = (stop, last, owner, end)
        del self.free[:consumed]
        return ranges, previous

    def release(self, ranges, owner, end):
        free = sorted([*self.free, *((first, last, owner, end) for first, last in ranges)])
        merged = []
        for interval in free:
            if merged and merged[-1][1] == interval[0] and merged[-1][2:] == interval[2:]:
                merged[-1] = (merged[-1][0], *interval[1:])
            else:
                merged.append(interval)
        self.free = merged


class Publications:
    """Keep consecutive publications as die/index arrays instead of Python pairs."""

    __slots__ = ('chunks', 'size')

    def __init__(self):
        self.chunks = []
        self.size = 0

    @classmethod
    def from_columns(cls, dies, indices):
        result = cls()
        if len(indices) == 1:
            result.extend_requests(int(dies[0]), indices)
            return result
        if len(indices):
            boundaries = np.r_[0, np.flatnonzero(dies[1:] != dies[:-1]) + 1, len(dies)]
            for first, last in zip(boundaries[:-1], boundaries[1:]):
                result.extend_requests(int(dies[first]), indices[first:last])
        return result

    def append(self, pair):
        die, index = pair
        if self.chunks and self.chunks[-1][0] == die and isinstance(self.chunks[-1][1], list):
            self.chunks[-1][1].append(index)
        else:
            self.chunks.append((die, [index]))
        self.size += 1

    def extend_requests(self, die, indices):
        values = np.asarray(indices, dtype=np.uint32)
        if len(values):
            self.chunks.append((int(die), values))
            self.size += len(values)

    def groups(self):
        for die, indices in self.chunks:
            yield die, np.asarray(indices, dtype=np.uint32)

    def __len__(self):
        return self.size

    def __iter__(self):
        for die, indices in self.chunks:
            for index in indices:
                yield die, int(index)


def publication_groups(node):
    publications = node['publications']
    if isinstance(publications, Publications):
        yield from publications.groups()
    elif publications:
        # Small synthetic graphs can still provide ordinary publication pairs.
        values = np.asarray(publications, dtype=np.uint64)
        yield from Publications.from_columns(values[:, 0], values[:, 1]).groups()
