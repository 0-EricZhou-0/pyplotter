default_description_name = "description.settings"

import abc
from typing import Iterable

class Parsers:
    def __init__(self) -> None:
        pass
    @abc.abstractmethod
    def parse(self, fp: Iterable): ...
