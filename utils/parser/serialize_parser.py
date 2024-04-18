from pyplotter.utils.logger import Logger

import logging
from typing import Final, Sequence
import re, copy

_statname_parser_compname = Logger().register_component(__file__)

default_separator: Final[str] = "-"

class SerializeParser:
    def __init__(self, separator=default_separator) -> None:
        self.__separator = separator
        self.__match_rules = []

        self.__current_recognize_dim = 0
        self.__ret_match = []
        self.__ret_aux_match = []

    def __register_at_dim(self, attribute_list: str | Sequence, dim_idx: int, regex: bool) -> None:
        assert dim_idx >= self.__current_recognize_dim
        assert dim_idx < len(self.__match_rules)
        if isinstance(attribute_list, str):
            attribute_list = [attribute_list]
        if regex and len(attribute_list) > 1:
            assert all([
                re.compile(attribute).groups == 0
                for attribute in attribute_list
            ]), "capturing groups are not allowed in regex mode"
        self.__match_rules[dim_idx] = \
            fr"""(?:^|{self.__separator})({"|".join([
                re.escape(attribute) if not regex else attribute
                for attribute in attribute_list
            ])})(?:$|{self.__separator})"""
        assert dim_idx >= self.__current_recognize_dim

    def register(self, attribute_list: str | Sequence, regex: bool=False) -> None:
        dim_idx = len(self.__match_rules)
        self.__match_rules.append(None)
        self.__register_at_dim(attribute_list, dim_idx, regex)
        assert self.__match_rules[dim_idx] is not None

    def reregister(self, attribute_list: str | Sequence, dim_idx: int, regex: bool=False) -> None:
        self.__register_at_dim(attribute_list, dim_idx, regex)

    def __recognize_once(self, pattern: str, composite_str: str) -> tuple[str, str | None, Sequence[str] | None]:
        rmatch = re.search(pattern, composite_str)
        if rmatch is not None:
            start_idx, end_idx = rmatch.start(), rmatch.end()
            match = rmatch.group(1)
            aux_matches = rmatch.groups()[1:]
            ret_str: str = f"""{composite_str[:start_idx]}{
                self.__separator if start_idx != 0 and end_idx != len(composite_str) else ""
            }{composite_str[end_idx:]}"""
            assert re.search(pattern, ret_str) is None
            return ret_str, match, None if len(aux_matches) == 0 else aux_matches
        return composite_str, None, None

    def recognize(self, composite_str: str, ndim: int=-1, persist: bool=False) -> tuple[str, Sequence[str | None], Sequence[Sequence[str]]]:
        dim_end = len(self.__match_rules) if ndim == -1 else \
            self.__current_recognize_dim + ndim
        assert dim_end > 0 and dim_end <= len(self.__match_rules)
        for match_rule in self.__match_rules[self.__current_recognize_dim:dim_end]:
            composite_str, match, aux_match = self.__recognize_once(match_rule, composite_str)
            self.__ret_match.append(match)
            self.__ret_aux_match.append(aux_match)
        self.__current_recognize_dim = dim_end
        ret = composite_str, copy.deepcopy(self.__ret_match), copy.deepcopy(self.__ret_aux_match)
        if not persist:
            self.clear_matches()
        return ret

    def clear_matches(self):
        self.__current_recognize_dim = 0
        self.__ret_match.clear()
        self.__ret_aux_match.clear()