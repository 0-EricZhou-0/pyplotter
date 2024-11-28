from __future__ import annotations

from pyplotter.utils.logger import Logger

import logging
from typing import TYPE_CHECKING, Any
if TYPE_CHECKING:
    from _typeshed import SupportsRead, SupportsWrite
import numpy.typing as npt

import numpy as np

_legacy_data_gen_compname = Logger().register_component(__file__)

def dump(settings_list: list[list[str]], data: npt.NDArray[Any], fp: SupportsWrite[str], verbose=False, float_precision=3, **kwargs) -> None:
    data_ndim = len(settings_list)
    data_dim_lens = [len(dim) for dim in settings_list]
    # TODO: write assert message here
    assert data_ndim == len(data.shape), f"{data_ndim} {len(data.shape)}"
    assert tuple(data_dim_lens) == tuple(data.shape), f"{data_dim_lens} {data.shape}"

    def format_element(d: Any, field_len: int=0) -> str:
        return_str: str = d if isinstance(d, str) else \
                          "None" if d is None else \
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

    fp.write(f"""{"|".join(settings_list[0])}""")
    if data_ndim == 1:
        fp.write(f"""\n\n{" ".join([format_element(d, data_field_len) for d in data[:]])}""")
    elif data_ndim == 2:
        for dim2_idx in range(data_dim_lens[1]):
            fp.write(f"\n\n{settings_list[1][dim2_idx]:{setting_field_len}}")
            fp.write(f"""\n{" ".join([format_element(d, data_field_len) for d in data[:, dim2_idx]])}""")
    elif data_ndim == 3:
        for dim2_idx in range(data_dim_lens[1]):
            fp.write(f"\n\n{settings_list[1][dim2_idx]:{setting_field_len}}")
            for dim3_idx in range(data_dim_lens[2]):
                fp.write(f"\n{settings_list[2][dim3_idx]:{setting_field_len}} ")
                fp.write(f"""{" ".join([format_element(d, data_field_len) for d in data[:, dim2_idx, dim3_idx]])}""")
    else:
        assert False, Logger().log(_legacy_data_gen_compname, logging.ERROR,
            f"Legacy data generator does not support input dimension > 3, get {data_ndim}")