from abc import ABCMeta, abstractmethod


class FaradaySource(metaclass=ABCMeta):
    def __init__(self):
        pass


class FaradayThinSource(FaradaySource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class FaradayThickSource(FaradaySource):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
