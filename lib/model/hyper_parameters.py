from dataclasses import dataclass


@dataclass
class HyperParameters(dict):
    def __set_item(self, key, item):
        raise TypeError('HyperParameters are frozen, cannot set values')

    def __getitem__(self, key: str):
        return getattr(self, key)
