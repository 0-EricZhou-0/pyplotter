from __future__ import annotations

from pyplotter.utils.logger import Logger

import logging
from typing import TYPE_CHECKING, Any, Callable, Sequence
if TYPE_CHECKING:
    from _typeshed import SupportsRead, SupportsWrite
import numpy.typing as npt

import numpy as np

_latex_table_gen_compname = Logger().register_component(__file__)

default_template = """\\begin{{tabular}}{{{column_style}}}
{content}
\\end{{tabular}}
"""

default_data_formatter: Callable[[Any], str] = lambda field: f"{field:.3f}"

__valid_column_styles = ["c"]

def dumps(settings_list: list[list[str]],
          data: npt.NDArray[Any],
          dimension_setting: tuple[Sequence[int] | None, Sequence[int] | None]=(None, None),
          float_precision=3,
          template: str=default_template,
          data_formatter: Callable[[Any], str]=default_data_formatter,
          column_style="c",
          verbose=False) -> str:
    data_ndim = len(settings_list)
    data_dim_lens = [len(dim) for dim in settings_list]
    assert data_ndim == len(data.shape)
    assert tuple(data_dim_lens) == tuple(data.shape), f"{data_dim_lens} {data.shape}"

    def format_element(d: Any, field_len: int=0) -> str:
        return_str: str = d if isinstance(d, str) else \
                          str(np.nan) if np.isnan(d) else \
                          str(np.inf) if np.isinf(d) else \
                          f"{d:.{float_precision}f}"
        return return_str if field_len == 0 else f"{return_str:>{field_len}s}"

    setting_field_len = max([
        len(setting)
        for settings in settings_list[1:] for setting in settings
    ]) if verbose else 0
    data_field_len = max([
        len(format_element(d))
        for _, d in np.ndenumerate(data)
    ]) if verbose else 0
    field_len = max(setting_field_len, data_field_len)

    assert column_style in __valid_column_styles
    column_list: str = ""
    content: str = ""
    if data_ndim == 1:
        raise NotImplementedError()
    elif data_ndim == 2:
        column_list = f"""|{"|".join([column_style for _ in range(data_dim_lens[0] + 1)])}|"""
        content = f"\\hline\n"
        content += f"""{"":{setting_field_len}s} & """
        content += f"""{" & ".join([format_element(s, field_len) for s in settings_list[0]])} \\\\"""
        content += "\\hline\n"
        for dim2_idx in range(data_dim_lens[1]):
            content += f"""{format_element(settings_list[1][dim2_idx], setting_field_len)} & """
            content += f"""{" & ".join([format_element(d, field_len) for d in data[:, dim2_idx]])} \\\\"""
            content += "\\hline\n"
        content = content.strip()
    else:
        assert False, Logger().log(_latex_table_gen_compname, logging.ERROR,
            f"Latex table generator does not support input dimension > 2, get {data_ndim}")
    return template.format(column_style=column_list, content=content)

def dump(settings_list: list[list[str]],
         data: npt.NDArray[Any],
         fp: SupportsWrite[str],
         dimension_setting: tuple[Sequence[int] | None, Sequence[int] | None]=(None, None),
         float_precision=3,
         template: str=default_template,
         data_formatter: Callable[[Any], str]=default_data_formatter,
         column_style="c",
         verbose=False) -> None:
    fp.write(dumps(
        settings_list, data,
        dimension_setting, float_precision, template, data_formatter, column_style,
        verbose
    ))
