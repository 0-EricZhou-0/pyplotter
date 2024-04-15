from __future__ import annotations

import pyplotter.utils.decorator as deco
import pyplotter.utils.env_variable as env

import os, logging

@deco.singleton
class Logger:
    """
    Wrapper of a two-level hierarchical logging.Logger
    """
    def __init__(self) -> None:
        module_name = __name__.split(".")[0]
        logging.basicConfig(filename=f"{module_name}.log", filemode="w", format="%(levelname)s - %(message)s")
        self.__default_logger: logging.Logger = logging.getLogger(module_name)
        self.__default_logger.setLevel(logging.DEBUG if env.is_debug() else logging.WARNING)
        self.__registered_logger_names = set()

    @property
    def default_logging_level(self) -> int:
        return self.__default_logger.level

    def __get_component_logger(self, component_name: str) -> logging.Logger | None:
        return self.__default_logger.getChild(component_name) if component_name in self.__registered_logger_names else None

    def __get_component_logger_or_default(self, component_name: str | None) -> logging.Logger:
        component_logger = None
        if component_name is not None:
            component_logger = self.__get_component_logger(component_name)
        return self.__default_logger if component_logger is None else component_logger

    def __register_component_logger(self, component_name: str, level: str | int | None) -> None:
        if component_name in self.__registered_logger_names:
            return
        self.__registered_logger_names.add(component_name)
        self.__default_logger.getChild(component_name).setLevel(level if level is not None else logging.NOTSET)

    def set_default_logging_level(self, level: str | int | None) -> int:
        self.__default_logger.setLevel(level if level is not None else logging.NOTSET)
        return self.__default_logger.level

    def set_component_logging_level(self, component_name: str, level: str | int | None) -> int:
        component_logger = self.__get_component_logger(component_name)
        assert component_logger is not None, self.log(_logger_compname, logging.ERROR,
                    f"Component {component_name} not registered")
        component_logger.setLevel(level if level is not None else logging.NOTSET)
        return component_logger.getEffectiveLevel()

    def __get_readable_name(self, component_name: str) -> str:
        """
        Get more human-readable name, interpreted from input component name (which is likely to
        be __file__ by design).

        Input Args:
            `component_name`: input name, likely to be __file__ of corresponding component

        Returns:
            more human-readable component name
        """
        component_name = os.path.basename(component_name).split(".")[0]
        component_name = component_name.replace("_", " ")
        if " " in component_name:
            component_name = " ".join([word.capitalize() for word in component_name.split()])
        else:
            split_idxs = [
                0,
                *[i for i, c in enumerate(component_name) if c.isupper()],
                len(component_name)
            ]
            component_name = " ".join([
                component_name[split_idxs[i]:split_idxs[i + 1]].capitalize()
                for i in range(len(split_idxs) - 1)
            ])
        return component_name

    def register_component(self, component_name: str, level: str | int | None=None, auto_readable: bool=True) -> str:
        if auto_readable:
            component_name = self.__get_readable_name(component_name)
        assert self.__get_component_logger(component_name) is None, self.log(_logger_compname, logging.ERROR,
                f"Component name {component_name} is registered twice")
        self.__register_component_logger(component_name, level)
        return component_name

    def get_component_logging_header(self, component_name: str) -> str:
        return f"[{component_name}]: "

    def component_should_log(self, component_name: str | None, level: int) -> bool:
        component_logger = None
        if component_name is not None:
            component_logger = self.__get_component_logger(component_name)
        component_logger = self.__default_logger if component_logger is None else component_logger
        return component_logger.isEnabledFor(level)

    def log(self, component_name: str | None, level: int, msg: str, *args, **kwargs) -> str:
        augmented_msg: str = f"{self.get_component_logging_header(component_name)}{msg}" if component_name is not None else msg
        logger: logging.Logger = self.__get_component_logger_or_default(component_name)
        logger.log(level, augmented_msg, *args, **kwargs)
        return augmented_msg

_logger_compname = Logger().register_component(__file__)
