from pathlib import Path

import numpy as np


def init(cache_type, **cache_args):
    if cache_type == 'numpy':
        cache_instance = NumpyCache.get_cache(**cache_args)
    elif cache_type == 'sql':
        # cache_instance = SqlCache.get_cache(**cache_args)
        raise NotImplementedError
    else:
        raise NotImplementedError(
            f'Cache of type {cache_type} is not implemented yet')
    return cache_instance


class Cache:
    cache_instance = None

    def __init__(self, grid_size):
        self.grid_size = grid_size

    @classmethod
    def get_cache(cls, **cache_args):
        if cls.cache_instance is not None:
            # sanity check
            return cls.cache_instance
        return cls(**cache_args)

    def is_done(self, *pos):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError

    def export(self, **kwargs):
        raise NotImplementedError

    def save_result(self, **kwargs):
        raise NotImplementedError

    def get_values(self):
        raise NotImplementedError


class NumpyCache(Cache):

    def __init__(self, grid_size):
        super().__init__(grid_size)
        self.values = np.zeros(self.grid_size)
        self._is_done = np.full(self.grid_size, False)

    def clear(self):
        self.values = np.zeros(self.grid_size)
        self._is_done = np.full(self.grid_size, False)

    def is_done(self, *pos):
        return self._is_done[pos]

    def export(self, export_path=None):
        if export_path:
            path = Path(export_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            # We do not care about race condition in parallel running.
            np.save(export_path, self.values)

    def save_result(self, value, *pos):
        if self.is_done(*pos):
            print(f'| [WARNING] cache[{pos}] is now changed from '
                  f'{self.values[pos]} to {value}.')
        self.values[pos] = value
        self._is_done[pos] = True

    def get_values(self):
        return self.values


class SqlCache(Cache):
    LANDSCAPE_SCHEMA = """
        landscape(
            id INTEGER PRIMARY KEY,
            pos_x INTEGER,
            pos_y INTEGER,
            loss REAL,
            datetime TEXT,
            landscape TEXT,
        )"""

    def clear(self):
        pass

    def is_done(self, *pos):
        pass

    def export(self, **kwargs):
        pass

    def save_result(self, **kwargs):
        pass

    def get_values(self):
        pass

    @classmethod
    def create_table(cls, conn):
        cursor = conn.cursor()
        cursor.execute(f"CREATE TABLE {cls.LANDSCAPE_SCHEMA}")
        conn.commit()
