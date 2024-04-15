import matplotlib as mpl
import matplotlib.axes as pa
import matplotlib.pyplot as plt

import numpy as np

def plot_bar(ax: pa.Axes):
    # fill in data and x_tick
    data_array = np.zeros((len(dim2), len(dim1), len(dim3)))
    x_tick_array = np.zeros((len(dim2), len(dim1)))
    for expr_grounp in sections[1:]:
    lines = expr_grounp.strip().split("\n")
    workload = lines[0].strip()
    workload_idx = dim2.index(workload)
    for line in lines[1:]:
        words = line.strip().split()
        measurement = " ".join(words[0:-len(dim1)]).strip()
        measurement_idx = dim3.index(measurement)
        data_array[workload_idx, :, measurement_idx] = np.array([float(data.strip()) / data_scale for data in words[-len(dim1):]])
        x_tick_array[workload_idx, :] = workload_idx * horiz_major_tick + (np.arange(len(dim1)) - (len(dim1) - 1) / 2) * bar_width

    agg_slice = np.sum(data_array, axis=2)
    if normalize_base is not None and len(normalize_base) != 0:
    normalize_base_idx = dim1.index(normalize_base)
    normalize_base_count = agg_slice[:, normalize_base_idx]
    if ymax is None:
    ymax = np.max(agg_slice) * 1.05

    for measurement_idx, measurement in enumerate(dim3):
    if measurement_idx == 0:
        bottom_slice = np.zeros(data_array[:, :, 0].shape)
    else:
        bottom_slice = np.sum(data_array[: , :, :measurement_idx], axis=2)
    for workload_idx, workload in enumerate(dim2):
        if normalize_base is None or len(normalize_base) == 0:
        normalize_divisor = 1
        else:
        normalize_divisor = normalize_base_count[workload_idx]
        to_plot_array = data_array[workload_idx, :, measurement_idx] / normalize_divisor
        bottom_plot_array = bottom_slice[workload_idx, :] / normalize_divisor
        ax.bar(x_tick_array[workload_idx, :], to_plot_array, color=color_arr[measurement_idx], bottom=bottom_plot_array, width=bar_width, edgecolor=hatch_color, hatch=hatch_arr[measurement_idx], label=measurement, zorder=3)
        ax.bar(x_tick_array[workload_idx, :], to_plot_array, color="none", bottom=bottom_plot_array, width=bar_width, edgecolor="#FFFFFF", linewidth=0.8, zorder=3)

    x_subticks = np.ravel(x_tick_array, order='f')
    x_subtick_labels = [setting[0].upper() if setting != "bytefs_cow" else "B'" for setting in dim1 for _ in dim2]
    x_minor_ticks = np.unique(np.hstack([x_subticks - bar_width / 2, x_subticks + bar_width / 2]))
    x_major_ticks = np.array([horiz_major_tick * (0.5 + x_major_tick_idx) for x_major_tick_idx in range(len(dim2) - 1)])

    ax.set_xticks(x_major_ticks)
    ax.set_xticklabels(["" for _ in x_major_ticks])
    ax.xaxis.set_minor_locator(FixedLocator(x_minor_ticks))
    for x_subtick, x_subtick_label in zip(x_subticks, x_subtick_labels):
    ax.text(x_subtick, ymin - (ymax - ymin) * 0.04, x_subtick_label, ha='center', va='top', fontsize=15)
    for x_tick, x_tick_label in zip(np.array([horiz_major_tick * x_tick_idx for x_tick_idx in range(len(dim2))]), dim2):
    ax.text(x_tick, ymin - (ymax - ymin) * 0.15, x_tick_label, ha='center', va='top', fontsize=20)
    ax.tick_params(which='major', width=1.6, length=9)
    ax.set_xlim([-horiz_margin * horiz_major_tick, (len(dim2) - 1 + horiz_margin) * horiz_major_tick])
    if ytick is not None:
    yticks = np.arange(ymin, ymax + 1, ytick)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(y) for y in yticks])
    ax.set_ylim([ymin, ymax])
    if normalize_base is None or len(normalize_base) == 0:
    if unit is None or len(unit) == 0:
        ax.set_ylabel(f"{ylabel}", fontsize=20)
    else:
        ax.set_ylabel(f"{ylabel} ({unit})", fontsize=20)
    else:
    for x_tick, data in zip(x_tick_array[:, normalize_base_idx], agg_slice[:, normalize_base_idx]):
        ax.text(x_tick + bar_width * 0.1, 1 + 0.02 * (ymax - ymin), f"{data:.2f} {unit}", ha="center", va="bottom", rotation=90, fontsize=14)
    ax.set_ylabel(f"{ylabel}", fontsize=20)
    ax.yaxis.grid(zorder=0)

    if force_ylim:
    ax.set_ylim([ymin, ymax])
    ymin, ymax = ax.get_ylim()

    for workload_idx, workload in enumerate(dim2):
    if normalize_base is None or len(normalize_base) == 0:
        normalize_divisor = 1
    else:
        normalize_divisor = normalize_base_count[workload_idx]
    to_plot_array = agg_slice[workload_idx, :] / normalize_divisor
    for setting_idx, to_plot_bar in enumerate(to_plot_array):
        if to_plot_bar > ymax:
        # ax0.text(x_tick_array[workload_idx, setting_idx] + bar_width * 0.1, ymax * 1.02, f"{to_plot_bar:.2f}x", ha="center", va="bottom", fontsize=13, rotation=90)
        ax.text(x_tick_array[workload_idx, setting_idx] + bar_width * 0.1, ymax * 0.98, f"{to_plot_bar:.2f}x", ha="center", va="top", fontsize=13, rotation=90, color="white")

    handles, labels = ax.get_legend_handles_labels()
    handles, labels = handles[0:len(handles):len(dim2)], labels[0:len(labels):len(dim2)]
    legend = ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.45, 1.4), ncol=len(labels))
    ax.hlines(0, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], zorder=9, color='black', linewidth=1)