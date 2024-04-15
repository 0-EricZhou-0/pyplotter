from __future__ import annotations

from pyplotter.utils.logger import Logger
import pyplotter.utils.colored_print as cp
import pyplotter.utils.decorator as deco

import enum, aenum, logging
from typing import Type

__unit_converter_compname: str = Logger().register_component(__file__)

class NameFormat(aenum.OrderedEnum):
    SHORT:       NameFormat = aenum.auto() # type: ignore
    DEFAULT:     NameFormat = SHORT
    SHORT_ALTER: NameFormat = aenum.auto() # type: ignore
    ABBR:        NameFormat = aenum.auto() # type: ignore
    LONG:        NameFormat = aenum.auto() # type: ignore

class WithEnumContainer:
    class Preset(aenum.IntEnum):
        @classmethod
        def add(cls, name: str) -> None:
            aenum.extend_enum(cls, name)
    def __init__(self, enum_cls: Preset) -> None:
        self.__forward_mapping: dict[WithEnumContainer.Preset, dict[NameFormat, str]] = {}
        self.__reverse_mapping: dict[str, WithEnumContainer.Preset] = {}
        self.__enum: WithEnumContainer.Preset = enum_cls
    def insert_key(self, name: Preset | str, representations: dict[NameFormat, str]) -> Preset:
        is_preset: bool = type(name) is not str
        displayed_name: str = name.name if is_preset else name # type: ignore
        assert NameFormat.DEFAULT in representations, Logger().log(
                __unit_converter_compname, logging.ERROR,
                f"Default name format {NameFormat.DEFAULT.name} for unit prefix {displayed_name} is not set"
        )
        default_name: str = representations[NameFormat.DEFAULT]
        assert default_name not in self.__reverse_mapping, [
            other_name := self.__enum(self.__reverse_mapping[default_name]).name,
            name_len := max(len(displayed_name), len(other_name)),
            this_representations := {str(k) : v for k, v in representations.items()},
            dup_representations := {
                str(k) : v
                for k, v in self.__forward_mapping[self.__reverse_mapping[default_name]].items()
            },
            Logger().log(
                __unit_converter_compname, logging.ERROR,
                f"Registering {displayed_name} that have same default representation ({NameFormat.DEFAULT}) as "
                f"{self.__class__.__name__}.{self.__enum.__name__}.{other_name}\n"
                f"  Registering {displayed_name:{name_len}s} ( {this_representations} )\n"
                f"  Duplicates  {other_name:{name_len}s} ( {dup_representations} )")
            ][-1]
        enum_name: WithEnumContainer.Preset
        if not is_preset:
            enum_key: str = name
            assert enum_key not in self.__enum.__members__, Logger().log(
                __unit_converter_compname, logging.ERROR,
                f"Item {enum_key} is already a member of {self.__enum.__name__}:\n"
                f"  {list(self.__enum.__members__.keys())}"
            )
            self.__enum.add(enum_key)
            enum_name = self.__enum.__members__[enum_key]
        else:
            enum_name = name
        self.__forward_mapping[enum_name] = representations
        self.__reverse_mapping[default_name] = enum_name
        return enum_name

    def __get_enum_name_repr(self, name: Preset | str) -> tuple[Preset, dict[NameFormat, str]]:
        enum_name: WithEnumContainer.Preset
        if type(name) is str:
            assert name in self.Preset.__members__, Logger().log(
                    __unit_converter_compname, logging.ERROR,
                    f"Trying to access non-existing key {name} from {self.Preset.__name__}:\n"
                    f"  {list(self.Preset.__members__.keys())}")
            enum_name = self.Preset.__members__[name]
        else:
            enum_name = name
        assert enum_name in self.__forward_mapping
        return enum_name, self.__forward_mapping[enum_name]

    def __get_name_or_sorted(self, name: Type[enum.Enum] | str, name_format: NameFormat, is_shorter: bool) -> str:
        name, representations = self.__get_enum_name_repr(name)
        if name_format in representations:
            return representations[name_format]
        sorted_name_format = sorted([
            format
            for format in NameFormat.__members__.values()
            if (format <= name_format if is_shorter else format >= name_format)
        ], reverse=is_shorter)
        for name_format_inst in sorted_name_format:
            if name_format_inst in representations:
                return representations[name_format_inst]
        return representations[NameFormat.DEFAULT]

    def get_name(self, name: Type[enum.Enum] | str, name_format: NameFormat | list[NameFormat]) -> str:
        name, representations = self.__get_enum_name_repr(name)
        name_format_list: list[NameFormat] = [name_format] if type(name_format) is NameFormat else \
                                             name_format
        for name_format_inst in name_format_list:
            if name_format_inst in representations:
                return representations[name_format_inst]
        return representations[NameFormat.DEFAULT]

    def get_name_or_shorter(self, name: Type[enum.Enum] | str, name_format: NameFormat) -> str:
        return self.__get_name_or_sorted(name, name_format, True)

    def get_name_or_longer(self, name: Type[enum.Enum] | str, name_format: NameFormat) -> str:
        return self.__get_name_or_sorted(name, name_format, False)

    def serialize(self, name: Preset) -> str:
        assert name in self.__forward_mapping
        return self.__forward_mapping[name][NameFormat.DEFAULT]

    def deserialize(self, name: str) -> Preset:
        assert name in self.__reverse_mapping
        return self.__reverse_mapping[name]

class UnitPrefixRepr(WithEnumContainer, metaclass=deco.Singleton):
    class Preset(WithEnumContainer.Preset):
        peta:    UnitPrefixRepr.Preset = aenum.auto() # type: ignore
        tera:    UnitPrefixRepr.Preset = aenum.auto() # type: ignore
        giga:    UnitPrefixRepr.Preset = aenum.auto() # type: ignore
        mega:    UnitPrefixRepr.Preset = aenum.auto() # type: ignore
        kilo:    UnitPrefixRepr.Preset = aenum.auto() # type: ignore
        none:    UnitPrefixRepr.Preset = aenum.auto() # type: ignore
        default: UnitPrefixRepr.Preset = none
        milli:   UnitPrefixRepr.Preset = aenum.auto() # type: ignore
        micro:   UnitPrefixRepr.Preset = aenum.auto() # type: ignore
        nano:    UnitPrefixRepr.Preset = aenum.auto() # type: ignore
        pico:    UnitPrefixRepr.Preset = aenum.auto() # type: ignore

    def __init__(self) -> None:
        super().__init__(UnitPrefixRepr.Preset)
        self.__scales: dict[UnitPrefixRepr.Preset, float] = {}
        self.add(UnitPrefixRepr.Preset.peta, {
            NameFormat.SHORT : "P",
            NameFormat.LONG : "peta",
        }, 1e15)
        self.add(UnitPrefixRepr.Preset.tera, {
            NameFormat.SHORT : "T",
            NameFormat.LONG : "tera",
        }, 1e12)
        self.add(UnitPrefixRepr.Preset.giga, {
            NameFormat.SHORT : "G",
            NameFormat.LONG : "giga",
        }, 1e9)
        self.add(UnitPrefixRepr.Preset.mega, {
            NameFormat.SHORT : "M",
            NameFormat.LONG : "mega",
        }, 1e6)
        self.add(UnitPrefixRepr.Preset.kilo, {
            NameFormat.SHORT : "k",
            NameFormat.LONG : "kilo",
            NameFormat.SHORT_ALTER : "K",
        }, 1e3)
        self.add(UnitPrefixRepr.Preset.none, {
            NameFormat.SHORT : "",
        }, 1)
        self.add(UnitPrefixRepr.Preset.milli, {
            NameFormat.SHORT : "m",
            NameFormat.LONG : "milli",
        }, 1e-3)
        self.add(UnitPrefixRepr.Preset.micro, {
            NameFormat.SHORT : "u",
            NameFormat.SHORT_ALTER : "μ",
            NameFormat.LONG : "micro",
        }, 1e-6)
        self.add(UnitPrefixRepr.Preset.nano, {
            NameFormat.SHORT : "n",
            NameFormat.LONG : "nano",
        }, 1e-9)
        self.add(UnitPrefixRepr.Preset.pico, {
            NameFormat.SHORT : "p",
            NameFormat.LONG : "pico",
        }, 1e-12)

    def add(self, name: UnitPrefixRepr.Preset | str, representations: dict[NameFormat, str], scale) -> None:
        enum_name = super().insert_key(name, representations)
        self.__scales[enum_name] = scale

    def get_scale(self, name: UnitPrefixRepr.Preset) -> float:
        return self.__scales[name]
# make singleton call override default class name
UnitPrefix = UnitPrefixRepr()

class UnitRepr(WithEnumContainer, metaclass=deco.Singleton):
    class Preset(WithEnumContainer.Preset):
        none:      UnitRepr.Preset = aenum.auto() # type: ignore
        sec:       UnitRepr.Preset = aenum.auto() # type: ignore
        second:    UnitRepr.Preset = sec
        b:         UnitRepr.Preset = aenum.auto() # type: ignore
        byte:      UnitRepr.Preset = b
        flop:      UnitRepr.Preset = aenum.auto() # type: ignore
        flops:     UnitRepr.Preset = flop
        op:        UnitRepr.Preset = aenum.auto() # type: ignore
        operation: UnitRepr.Preset = op

    def __init__(self) -> None:
        super().__init__(UnitRepr.Preset)
        self.add(UnitRepr.Preset.none, {
            NameFormat.SHORT: ""
        })
        self.add(UnitRepr.Preset.sec, {
            NameFormat.SHORT: "s",
            NameFormat.ABBR: "sec",
            NameFormat.LONG: "second",
        })
        self.add(UnitRepr.Preset.b, {
            NameFormat.SHORT: "b",
            NameFormat.ABBR: "sec",
            NameFormat.LONG: "byte",
        })
        self.add(UnitRepr.Preset.flop, {
            NameFormat.SHORT: "flop",
            NameFormat.ABBR: "sec",
            NameFormat.LONG: "byte",
        })

    def add(self, name: UnitRepr.Preset | str, representations: dict[NameFormat, str]) -> None:
        super().insert_key(name, representations)
# make singleton call override default class name
Unit = UnitRepr()

class PrefixedUnit():
    def __init__(self, unit: UnitRepr, unit_prefix: UnitPrefixRepr) -> None:
        pass
    def __mul__(self, other):
        pass
    def __rmul__(self, other):
        pass

class CompositePrefixedUnit:
    pass
