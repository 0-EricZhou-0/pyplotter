from __future__ import annotations

from pyplotter.utils.logger import Logger
import pyplotter.utils.colored_print as cp
import pyplotter.utils.generic as gutils

import json, logging, enum, itertools, copy, ast
from typing import TYPE_CHECKING, Any, Callable, Final, Sequence, Generator
if TYPE_CHECKING:
    from _typeshed import SupportsRead, SupportsWrite
import numpy.typing as npt

import numpy as np

_multidim_parser_compname = Logger().register_component(__file__)

class MultiDimDataRepresentation():
    sort_rule_type: Final = Callable[[list[str]], list[str]]
    slice_index_type: Final = slice | Sequence[str] | str | None

    @classmethod
    def check_json_format(cls, json: dict) -> None:
        json_property = "dim_names"
        assert json_property in json
        assert isinstance(json[json_property], Sequence)
        assert len(json[json_property]) > 0

        json_property = "setting_names"
        assert json_property in json
        assert isinstance(json[json_property], list)
        assert len(json[json_property]) == len(json["dim_names"])
        assert all([isinstance(arr, list)] for arr in json[json_property])
        assert all([len(arr) > 0 for arr in json[json_property]])

        # json_property = f"{MultiDimDataRepresentation.last_dim_name}_unit"
        # assert json_property in json
        # assert type(json[json_property] is list)
        # assert len(json[json_property]) == len(json["setting_names"][-1])

        json_property = "data"
        assert json_property in json

    # python is so garbage it does not have a nice way to overload functions, this is work around
    @classmethod
    def from_dim_names(cls, dim_names: Sequence[str]) -> MultiDimDataRepresentation:
        return cls(dim_names=dim_names)

    @classmethod
    def from_input_file(cls, fp: SupportsRead[str | bytes], **kwargs) -> MultiDimDataRepresentation:
        return cls(fp=fp, **kwargs)

    def __init__(self, dim_names=None, fp=None, **kwargs) -> None:
        if dim_names is not None and fp is None:
            self.__init_from_dim_names(dim_names)
        elif dim_names is None and fp is not None:
            self.__init_from_input_file(fp, **kwargs)
        else:
            assert False, Logger().log(_multidim_parser_compname, logging.ERROR,
                "Unsupported init method"
            )

    def __init_from_dim_names(self, dim_names: Sequence[str]) -> None:
        self.__ndim: int = len(dim_names)
        assert self.__ndim > 0, Logger().log(_multidim_parser_compname, logging.ERROR,
                f"There should be at least one dimension exist")
        self.__dim_names: list[str] = list(dim_names)
        self.__setting_names: list[dict[str, int]] = [{} for _ in self.__dim_names]
        self.__setting_sort_rules: list[None | MultiDimDataRepresentation.sort_rule_type] = [
            None for _ in self.__dim_names
        ]
        self.__data: dict[tuple[int, ...], Any] = dict()

    def __init_from_input_file(self, fp: SupportsRead[str | bytes], **kwargs) -> None:
        self.load(fp, **kwargs)

    def get_dim_names(self) -> list[str]:
        return copy.deepcopy(self.__dim_names)

    def get_setting_names(self) -> list[dict[str, int]]:
        return copy.deepcopy(self.__setting_names)

    def __get_int_idx_default(self, str_idx: str, dim_idx: int) -> int:
        dim_name_dict: dict[str, int] = self.__setting_names[dim_idx]
        return dim_name_dict[str_idx] if str_idx in dim_name_dict else -1

    def __get_int_idx_list_default(self, str_idx_list: Sequence[str]) -> tuple[int, ...]:
        return tuple(
            self.__get_int_idx_default(str_idx, dim_idx)
            for dim_idx, str_idx in enumerate(str_idx_list)
        )

    def __get_int_idx_list(self, str_idx_list: Sequence[str], create: bool=False) -> tuple[int, ...]:
        assert len(str_idx_list) == self.__ndim
        int_idx_list: tuple[int, ...] = self.__get_int_idx_list_default(str_idx_list)
        if int_idx_list not in self.__data and create:
            for dim_idx, str_idx in enumerate(str_idx_list):
                if str_idx not in self.__setting_names[dim_idx]:
                    int_idx: int = len(self.__setting_names[dim_idx])
                    self.__setting_names[dim_idx][str_idx] = int_idx
            return self.__get_int_idx_list_default(str_idx_list)
        return int_idx_list

    def __get_data(self, int_idx_list: tuple[int, ...]) -> Any:
        return self.__data[int_idx_list] if int_idx_list in self.__data else None

    def insert_data(self, str_idx_list: Sequence[str], data: Any,
                    modify_rule: Callable[[Any, Any], Any]=lambda orig, new : new) -> Any:
        if data is None:
            return self.erase_data(str_idx_list)
        int_idx_list: tuple[int, ...] = self.__get_int_idx_list(str_idx_list, create=True)
        tuple_idx = tuple(int_idx_list)
        orig = None if tuple_idx not in self.__data else self.__data[tuple_idx]
        ret = modify_rule(orig, data)
        self.__data[tuple_idx] = ret
        return ret

    def insert_append_data(self, str_idx_list: Sequence[str], data: Any) -> Any:
        if self.get_data(str_idx_list) is None:
            self.insert_data(str_idx_list, [])
        return self.insert_data(str_idx_list, data, lambda orig, new : orig + [new])

    def get_data(self, str_idx_list: Sequence[str]) -> Any:
        return self.__get_data(self.__get_int_idx_list(str_idx_list))

    def __parse_item_index_dim(self, dim_idx: int,
                               key: MultiDimDataRepresentation.slice_index_type) -> tuple[int, ...]:
        try:
            if key is None:
                return tuple(range(len(self.__setting_names[dim_idx])))
            elif isinstance(key, str):
                return (self.__get_int_idx_default(key, dim_idx), )
            elif isinstance(key, slice):
                if key.start is not None or key.stop is not None:
                    raise NotImplementedError(f"Slice start and stop {(key.start, key.stop)} not supported, accepts (None, None) only")
                if key is None or key.step == 1:
                    return tuple(range(len(self.__setting_names[dim_idx])))
                elif key.step == -1:
                    return tuple(reversed(range(len(self.__setting_names[dim_idx]))))
                else:
                    raise NotImplementedError(f"Slice step {key.step} not supported, accepts 1 or -1 only")
            elif isinstance(key, Sequence):
                return tuple(self.__get_int_idx_default(k, dim_idx) for k in key)
            else:
                raise NotImplementedError(f"Not supported index type {type(key)}")
        except Exception as e:
            assert False, Logger().log(_multidim_parser_compname, logging.ERROR,
                f"Indexing error: {e}"
            )

    def __getitem__(self, keys: MultiDimDataRepresentation.slice_index_type | Sequence[MultiDimDataRepresentation.slice_index_type]) -> Any:
        raise NotImplementedError()
        if isinstance(keys, slice) or isinstance(keys, str) or \
           (isinstance(keys, Sequence) and len(keys) > 0 and not isinstance(keys[0], Sequence)):
            # single dim
            pass
        else:
            if len(keys) != self.__ndim:
                keys = [*keys, *[None for _ in range(self.__ndim - len(keys))]]
            item_int_idxs: tuple[int, ...] = tuple(
                self.__parse_item_index_dim(dim_idx, key)
                for dim_idx, key in enumerate(keys)
            )
        dim_lens = tuple(len(item_int_idx) for item_int_idx in item_int_idxs)
        return_nparr = np.zeros(dim_lens, dtype=object)
        print(keys, list(item_int_idxs), return_nparr.shape)
        for idx_list in itertools.product(*[range(n) for n in override_dim_lens]):
            override_idx_list: tuple[int, ...] = tuple(orig_idx if not override else idx_list[dim_idx]
                for dim_idx, (orig_idx, override) in enumerate(zip(int_idx_list, override_dims))
            )
            return_nparr[idx_list] = self.__get_data(override_idx_list)
        print(list(item_int_idxs))
        return None

    def get_data_slice(self, str_idx_list: list[str | None] | tuple[str | None, ...]) -> npt.NDArray[Any]:
        override_dims: list[bool] = [str_idx == None for str_idx in str_idx_list]
        override_dim_lens: list[int] = [
            len(self.__setting_names[override_dim_idx])
            for override_dim_idx, override in enumerate(override_dims) if override
        ]
        str_idx_list_modified: list[str] = [str_idx if str_idx is not None else "" for str_idx in str_idx_list]
        int_idx_list: tuple[int, ...] = self.__get_int_idx_list(str_idx_list_modified)
        return_nparr = np.zeros(override_dim_lens, dtype=object)
        for idx_list in itertools.product(*[range(n) for n in override_dim_lens]):
            override_idx_list: tuple[int, ...] = tuple(orig_idx if not override else idx_list[dim_idx]
                for dim_idx, (orig_idx, override) in enumerate(zip(int_idx_list, override_dims))
            )
            return_nparr[idx_list] = self.__get_data(override_idx_list)
        return return_nparr

    def get_setting_name_slice(self, str_idx_list: list[str | None] | tuple[str | None, ...]) -> list[list[str]]:
        override_dims: list[bool] = [str_idx == None for str_idx in str_idx_list]
        return [
            list(self.__setting_names[dim_idx].keys())
            for dim_idx, override in enumerate(override_dims)
            if override
        ]

    def erase_data(self, str_idx_list: Sequence[str]) -> Any:
        int_idx_list: tuple[int, ...] = self.__get_int_idx_list(str_idx_list)
        return self.__data.pop(int_idx_list, None)

    def create_output_dict(self) -> dict:
        output_dict = dict()
        output_dict["dim_names"] = self.__dim_names
        output_dict["setting_names"] = self.__setting_names
        output_dict["data"] = { str(k): v for k, v in self.__data.items() }
        MultiDimDataRepresentation.check_json_format(output_dict)
        return output_dict

    def load(self, fp: SupportsRead[str | bytes], **kwargs) -> None:
        input_dict: dict = json.load(fp, **kwargs)
        MultiDimDataRepresentation.check_json_format(input_dict)
        self.__init_from_dim_names(input_dict["dim_names"])
        self.__setting_names: list[dict[str, int]] = input_dict["setting_names"]
        self.__data: dict[tuple[int, ...], Any] = dict()
        for k, v in input_dict["data"].items():
            self.__data[ast.literal_eval(k)] = v

    def dump(self, fp: SupportsWrite[str], **kwargs) -> None:
        json.dump(self.create_output_dict(), fp, **kwargs)

    def set_sort_rules(self, dim: str | int,
                       sort_rules = sort_rule_type | Sequence[None | sort_rule_type]) -> None:
        dim_idx: int
        if isinstance(dim, int):
            dim_idx = dim
        else:
            assert dim in self.__dim_names
            dim_idx = self.__dim_names.index(dim)
        if not isinstance(sort_rules, Sequence):
            sort_rules = [sort_rules]
        nrules = len(sort_rules) # type: ignore
        assert dim_idx + nrules <= len(self.__setting_sort_rules)
        self.__setting_sort_rules[dim_idx:dim_idx+nrules] = sort_rules[:] # type: ignore

    def sort(self) -> None:
        modification_rules: list[list[int]] = [[] for _ in range(self.__ndim)]
        for dim_idx, sort_rule in enumerate(self.__setting_sort_rules):
            if sort_rule is not None:
                orig_settings_list = list(self.__setting_names[dim_idx].keys())
                sorted_settings_list = list(sort_rule(orig_settings_list))
                assert [
                        sym_diff := set(orig_settings_list) ^ set(sorted_settings_list),
                        len(sym_diff) == 0
                    ][-1], Logger().log(_multidim_parser_compname, logging.ERROR, [
                        sort_rule_src := gutils.get_source_location(sort_rule),
                        f"In dimension {self.__dim_names[dim_idx]} (dim {dim_idx}), "
                        f"sort rule changes dimension content\n"
                        f"  Orig:   {orig_settings_list}\n"
                        f"  Sorted: {sorted_settings_list}\n"
                        f"  Symmetric Difference: {sym_diff}\n"
                        f"  Sort rule defined at {sort_rule_src if sort_rule_src is not None else 'Unable to find source'}"
                    ][-1]
                )
                modification_rules[dim_idx] = [
                    sorted_settings_list.index(setting)
                    for setting in orig_settings_list
                ]
                self.__setting_names
                self.__setting_names[dim_idx] = {
                    setting_name : setting_idx
                    for setting_idx, setting_name in enumerate(sorted_settings_list)
                }
        sorted_data: dict[tuple[int, ...], Any] = dict()
        for k, v in self.__data.items():
            sorted_data[tuple(
                orig_val if len(modification_rules[dim_idx]) == 0 else modification_rules[dim_idx][orig_val]
                for dim_idx, orig_val in enumerate(k)
            )] = v
        self.__data = sorted_data

    class MergeRules(enum.IntEnum):
        PRIORITIZE_THIS = 0
        PRIORITIZE_OTHER = 1
        ERROR_ON_CONFLICT = 2

    def merge(self, other: MultiDimDataRepresentation, merge_rule: MergeRules=MergeRules.PRIORITIZE_THIS) -> None:
        assert self.__ndim == other.__ndim, Logger().log(_multidim_parser_compname, logging.ERROR,
                f"Dimension name length mismatch Expected length: {self.__ndim} Actual length: {other.__ndim}")
        other_str_idx_dict_view: list[list[str]] = [
            list(str_dict.keys())
            for str_dict in other.__setting_names
        ]
        for other_int_idx_list, other_data in other.__data.items():
            other_str_idx_list: list[str] = [
                other_str_idx_dict_view[dim_idx][other_int_idx]
                for dim_idx, other_int_idx in enumerate(other_int_idx_list)
            ]
            this_data = self.get_data(other_str_idx_list)
            if this_data is not None and this_data != other_data:
                log_level: int = logging.ERROR if merge_rule == MultiDimDataRepresentation.MergeRules.ERROR_ON_CONFLICT else \
                                 logging.WARNING
                cp.lprintf(log_level, Logger().log(_multidim_parser_compname, log_level,
                        f"Merge conflit on key {other_str_idx_list} This data: <{this_data}> Merging data: <{other_data}>"
                ))
                assert log_level != logging.ERROR
                if merge_rule is not MultiDimDataRepresentation.MergeRules.PRIORITIZE_THIS:
                    self.insert_data(other_str_idx_list, other_data)
