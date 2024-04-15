import os, random, itertools
import matplotlib.pyplot as plt

colors_light1   = ["#ff796c", "#fac205", "#95d0fc", "#96f97b"]
colors_light2   = ["lightgrey", "#96f97b", "cyan", "#fc9096"]
colors_light3   = ["#bef7ff", "#b2ecff", "#a6e2ff", "#9ad7ff", "#8eccff", "#82c2ff", "#75b7ff", "#69acff", "#5da1ff", "#5197ff", "#458cff"]
colors_light4   = ["#a7e1ab", "#8cd892", "#72cf78", "#57c65f", "#3fba48", "#36a03d", "#2d8533", "#246a29", "#1b501e", "#123514", "#0a1a0a"]
colors_light5   = ["#ff796c", "#a4d46c", "#fca45c", "#95dbd0"]
colors_light6   = ["#638a66", "#a2bda2", "#c0e1d9", "#f0ebd6", "#f1cbb5", "#d2977c"]
colors_light7   = ["#333c42", "#316658", "#5ea69c", "#c2cfa2", "#a4799e", "#706690"]
colors_light8   = ["#ffba61", "#f74c0c", "#9200e0", "#3892fa", "#30ff00"]
colors_light9 = ["#123514", "#246a29", "#36a03d", "#57c65f", "#8cd892"]

colors_dark1    = ["brown", "royalblue", "peru", "forestgreen", "purple"]
colors_dark2    = ["#1e3d69", "#7d4282", "#bc5299", "#c83e6a", "#de7015", "#f4a300"]
colors_dark3    = ["#0094ff", "#008d00", "#ff9200", "#cfb99e", "#e5086a"]
colors_dark3_2  = ["#0094ff", "#008d00", "#ff9200", "#b39b7f", "#e5086a"]
colors_dark3_3  = ["#b39b7f", "#008d00", "#ff9200", "#0094ff", "#e5086a"]
colors_dark4    = ["#0d5b26", "#c94733", "#fddf8b", "#3fab47", "#52b9d8", "#2e5fa1"]
colors_dark5    = ["#3a1b19", "#7b595e", "#c7a085", "#fcf0e1", "#c94737"]
colors_dark5_2  = ["#3a1b19", "#7b595e", "#c7a085", "#c9bfb1", "#c94737"]
colors_dark6    = ["#1e1a22", "#67a4ba", "#4e4c72", "#b94e5e", "#d389a1"]
colors_dark7    = ["#2f2321", "#aa4f23", "#f8ebdc", "#fed875", "#6c627a"]
colors_dark7_2  = ["#2f2321", "#aa4f23", "#c9bfb1", "#e3c168", "#6c627a"]
colors_dark8    = ["#211a3e", "#453370", "#a597b6", "#fef3e8", "#d06c9d"]
colors_dark8_2  = ["#211a3e", "#453370", "#a597b6", "#c9bfb1", "#d06c9d"]
colors_dark9    = ["#33395b", "#5d74a2", "#c4d8f2", "#f2e8e3", "#7c282b"]
colors_dark9_2  = ["#33395b", "#5d74a2", "#adc0d9", "#d4cac5", "#7c282b"]
colors_dark10   = ["#c1272d", "#0000a7", "#eecc16", "#008176", "#b3b3b3"]
colors_dark11   = ["#ca6bf2", "#5568d9", "#ffffff", "#ffffff", "#ffffff"]

colors_yoasobi_1 = ["#604871", "#f45973", "#33718a", "#aa8a93", "#3c4f71"]

colors_roller_1 = ["#0072B2", "#57B4E9", "#CDCDCD"]
colors_roller_2 = ["#57B4E9", "#019E73", "#E69F00", "#0072B2", "#000000"]
colors_roller_3 = ["#522222", "#57B4E9", "#019E73", "#0072B2", "#E69F00"]
colors_roller_4 = ["#57B4E9", "#019E73", "#E69F00", "#0072B2", "#B21000"]
colors_roller_5 = ["#57B4E9", "#019E73", "#E69F00", "#0072B2", "#B21000", "#5B0680"]
colors_roller_6 = ["#57B4E9", "#019E73", "#E69F00", "#B21000", "#5B0680", "#0072B2"]
colors_roller_7 = ["#0072B2", "#019E73", "#E69F00", "#B21000", "#57B4E9"]
colors_roller_8 = ["#0072B2", "#01BE93", "#E69F00", "#B21000", "#57B4E9"]
colors_roller_9 = ["#453370", "#0072B2", "#57B4E9", "#CDCDCD"]
colors_roller_10 = ["#CDCDCD", "#57B4E9", "#0072B2", "#453370"]

colors_test  = random.sample(color_arr:=[
    f"#{r}{g}{b}"
    for r, g, b in itertools.product([
        f"{x:02x}" for x in range(0, 0xFF, 0x40)
    ], repeat=3)], len(color_arr)
)

colors_custom1 = ["#2BC233", "#E67009", "#B21000", "#2489FF"]

import math
import numpy as np
def get_data_scale(unit: str) -> float :
    metric_prefix = ""
    rectify_ratio = [1000, 1000]
    if unit.endswith("ib") or unit.endswith("iB"):
        metric_prefix = unit[:-2]
        rectify_ratio = [1000, 1024]
    elif unit.endswith("b") or unit.endswith("B"):
        metric_prefix = unit[:-1]
    elif unit.endswith("s"):
        metric_prefix = unit[:-1]
    elif unit.endswith("sec"):
        metric_prefix = unit[:-3].strip()

    data_scale_dict = {
        "m" : 1e-3, "u" : 1e-6, "n" : 1e-9, "p" : 1e-12,
        ""  : 1e0,
        "k" : 1e3 , "M" : 1e6 , "G" : 1e9 , "T" : 1e12,
    }

    if metric_prefix in data_scale_dict:
        data_scale = data_scale_dict[metric_prefix]
        assert len(rectify_ratio) == 2 and not np.isclose(rectify_ratio[0], 1)
        rectified_data_scale = math.pow(rectify_ratio[1], math.log(data_scale, rectify_ratio[0]))
        return 1 / rectified_data_scale
    print(f"Unit <{unit}> not indexed in data scale catalog")

class FigurePresets:
    @staticmethod
    def apply_default_style():
        preset_file = os.path.abspath(os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "matplotlib_style_sheets",
            "default.mplstyle"
        ))
        assert os.path.isfile(preset_file)
        plt.style.use(preset_file)
