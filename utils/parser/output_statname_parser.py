from pyplotter.utils.logger import Logger

import logging, abc

__statname_parser_compname = Logger().register_component(__file__)

class OutputStatNameParser:
    def __init__(self) -> None:
        self.__settings = []
        self.__workloads = []
    @abc.abstractmethod
    def recognize(stat_filename: str) -> None: ...

    def register_setting(setting: str):
        pass
    def register_workload(workload: str):
        pass