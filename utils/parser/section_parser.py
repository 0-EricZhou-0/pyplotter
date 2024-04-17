from __future__ import annotations

from pyplotter.utils.logger import Logger
import pyplotter.utils.multidim_repr as mdr
import pyplotter.utils.parser.generic_parser as gp

import logging, abc, re, enum
from typing import Final, Callable, Iterable, Any

__section_parser_compname: str = Logger().register_component(__file__)

class SectionParsers(gp.Parsers):
    default_section_header: Final[str] = "Section:"
    def __init__(self, section_begin_str: str | None=default_section_header) -> None:
        self.__section_begin_str: str | None = section_begin_str
        self.__section_parsers: dict[str, SectionParser] = dict()
        self.__current_section: str = ""

    def register_section_parser(self, section_parser: SectionParser):
        section_name: str = section_parser.get_section_name()
        self.__section_parsers[section_name] = section_parser
        if self.__section_begin_str is None:
            assert len(self.__section_parsers) == 1, Logger().log(__section_parser_compname, logging.ERROR,
                "Section parsers with no section begin str can only register one section parser")
            self.__current_section = section_name

    def parse(self, fp: Iterable) -> None:
        for line in fp:
            line: str = line.strip()
            # switching section
            if self.__section_begin_str is not None and line.startswith(self.__section_begin_str):
                new_section = line[len(self.__section_begin_str):].strip()
                if new_section in self.__section_parsers.keys():
                    if self.__current_section == new_section:
                        continue
                    if self.__current_section in self.__section_parsers:
                        self.__section_parsers[self.__current_section].on_section_end()
                    self.__section_parsers[new_section].on_section_begin()
            if self.__current_section in self.__section_parsers:
                self.__section_parsers[self.__current_section].on_section_line(line)
        for parser in self.__section_parsers.values():
            parser.on_parser_finish()

    def get_section_parsers(self) -> list[SectionParser]:
        return list(self.__section_parsers.values())

class SectionParser():
    def __init__(self, section_name: str) -> None:
        self.__section_name: str = section_name.strip()
        # warnif len(self.__section_name) != len(section_name)

    def get_section_name(self) -> str:
        return self.__section_name

    @abc.abstractmethod
    def on_section_begin(self) -> None: ...

    @abc.abstractmethod
    def on_section_line(self, line: str) -> None: ...

    @abc.abstractmethod
    def on_section_end(self) -> None: ...

    @abc.abstractmethod
    def on_parser_finish(self) -> None: ...

    def __str__(self) -> str: ...

class CommentsParser(SectionParser):
    def __init__(self) -> None:
        super().__init__("Comments")

    def on_section_begin(self) -> None:
        pass

    def on_section_line(self, line: str) -> None:
        pass

    def on_section_end(self) -> None:
        pass

    def on_parser_finish(self) -> None:
        pass

    def __str__(self) -> str:
        return ""

class CalculatedDimensionParser(SectionParser):
    class Statistic(enum.Enum):
        sum = enum.auto()
        total = sum
        average = enum.auto()
        avg = average
        mean = average
        variance = enum.auto()
        var = variance
        standard_deviation = enum.auto()
        std = standard_deviation
    def __init__(self, data_representation: mdr.NamedArray) -> None:
        super().__init__("Calculated Dimension")
        self.__data_representation: mdr.NamedArray = data_representation

    def on_section_begin(self):
        pass

    def on_section_line(self, line: str) -> None:
        pass

class SettingsParser(SectionParser):
    def __init__(self, delimiters: list[str]=[":"], comment_starts: list[str]=["#"]) -> None:
        super().__init__("Settings")
        self.__delimiters: list[str] = delimiters
        self.__comment_starts: list[str] = comment_starts
        self.__setting_dict: dict[str, str] = dict()

    def on_section_begin(self) -> None:
        pass

    def on_section_line(self, line: str) -> None:
        # skip comment
        if (any([line.startswith(comment_start) for comment_start in self.__comment_starts])):
            return
        # parse setting according to delimiter
        kv_list: list[str] = re.split(" | ".join(self.__setting_dict), line)
        if len(kv_list) != 2:
            raise ValueError((
                Logger().get_component_logging_header(__section_parser_compname),
                f"Line <{line}> is not a valid setting. Under delimiters {self.__delimiters}"
            ))
        self.__setting_dict[kv_list[0].strip()] = kv_list[1].strip()

    def on_section_end(self) -> None:
        pass

    def on_parser_finish(self) -> None:
        pass

def default_line_parser(line: str) -> list[tuple[str, Any]]:
    result = re.match(r"(?:.*:)?\s*([^:]*):\s*(.*)\s*", line)
    if result is not None and len(result.groups()) >= 2:
        measurement_name = " ".join(
            result.group(1).replace("(", " ").replace(")", " ").strip().split()) \
            .replace(" ", "_").replace("-", "_").lower()
        measurement_val = [
            float(d)
            for d in re.findall(r'\d+\.?\d*', result.group(2))
        ]
        # print(measurement_name, measurement_val)
        if len(measurement_val) == 0:
            return []
        elif len(measurement_val) == 1:
            return [(measurement_name, measurement_val[0])]
        else:
            return [(measurement_name, measurement_val)]
    return []

def default_final_data_parser(data: dict[str, Any]) -> dict[str, Any]:
    return data

class StatParser(SectionParser):
    def __init__(self,
                 line_parser: Callable[[str], list[tuple[str, Any]]]=default_line_parser,
                 final_data_parser: Callable[[dict[str, Any]], dict[str, Any]]=default_final_data_parser) -> None:
        super().__init__("Stats")
        self.__dataset: dict[str, Any] = dict()
        self.__line_parser: Callable[[str], list[tuple[str, Any]]] = line_parser
        self.__final_data_parser: Callable[[dict[str, Any]], dict[str, Any]] = final_data_parser

    def on_section_begin(self) -> None:
        pass

    def on_section_line(self, line: str) -> None:
        stats: list[tuple[str, Any]] = self.__line_parser(line)
        for measurement_name, measurement_val in stats:
            self.__dataset[measurement_name] = measurement_val

    def on_section_end(self) -> None:
        pass

    def on_parser_finish(self) -> None:
        self.__dataset = self.__final_data_parser(self.__dataset)

    def get_dataset(self) -> dict[str, Any]:
        return self.__dataset