# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from matplotlib.ticker import FixedLocator

from figure_utils.figure_utils import *

from matplotlib.cm import get_cmap

color_arr = colors_roller_9
hatch_color = "white"
hatch_arr = ["", "", "", "", "", ""]

# plot font size options
plt.rc("font", size=15)
plt.rc("axes", titlesize=30)
plt.rc("xtick", labelsize=18)
plt.rc("ytick", labelsize=18)
plt.rc("legend", fontsize=18)
plt.rc("hatch", color="white")
mpl.rcParams["hatch.linewidth"] = 1.8
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams.update({'font.size': 16})
mpl.rcParams.update({'font.family': 'serif'})

fig, ax0 = plt.subplots(1, 1, figsize=(12, 2))
plt.subplots_adjust(top = 1.02, bottom=0.01, hspace=0.6, wspace=0.20)

bar_width = 0.085
horiz_margin = 0.6
horiz_major_tick = 0.7

force_ylim = False
# ===============================================
io_file_name = f"breakdown"
normalize_base = None
unit = None
ylabel = "#Accesses"
force_ylim = False
# ===============================================

data_file = f"{io_file_name}.txt"
fig_file = f"{io_file_name}.pdf"

all_data_arr = np.zeros(0)
try:
  with open(data_file, "r") as f:
    lines = f.read()
except Exception as e:
  exit(1)


ymin, ymax, ytick = 0, 15, 2
ymin, ymax, ytick = 0, 2.5, 0.5
ymin, ymax, ytick = 0, None, None
data_scale = 1
if unit is not None:
  if unit.lower() == "b":
    data_scale = 1
  elif unit.lower() == "kb":
    data_scale = 1024
  elif unit.lower() == "mb":
    data_scale = 1024 ** 2
  elif unit.lower() == "gb":
    data_scale = 1024 ** 3

dim0 = []
dim1 = []
dim2 = []
dim3 = []

# get settings, workloads, and measurements name
sections = [section for section in lines.strip().split("\n\n") if len(section.strip()) != 1]
dim1 = [setting.strip() for setting in sections[0].split()]
for expr_grounp in sections[1:]:
  lines = expr_grounp.strip().split("\n")
  workload = lines[0].strip()
  dim2.append(workload)
  dim3 = [" ".join(line.strip().split()[0:-len(dim1)]).strip() for line in lines[1:]]

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
    ax0.bar(x_tick_array[workload_idx, :], to_plot_array, color=color_arr[measurement_idx], bottom=bottom_plot_array, width=bar_width, edgecolor=hatch_color, hatch=hatch_arr[measurement_idx], label=measurement, zorder=3)
    ax0.bar(x_tick_array[workload_idx, :], to_plot_array, color="none", bottom=bottom_plot_array, width=bar_width, edgecolor="#FFFFFF", linewidth=0.8, zorder=3)

x_subticks = np.ravel(x_tick_array, order='f')
x_subtick_labels = [setting[0].upper() if setting != "bytefs_cow" else "B'" for setting in dim1 for _ in dim2]
x_minor_ticks = np.unique(np.hstack([x_subticks - bar_width / 2, x_subticks + bar_width / 2]))
x_major_ticks = np.array([horiz_major_tick * (0.5 + x_major_tick_idx) for x_major_tick_idx in range(len(dim2) - 1)])

ax0.set_xticks(x_major_ticks)
ax0.set_xticklabels(["" for _ in x_major_ticks])
ax0.xaxis.set_minor_locator(FixedLocator(x_minor_ticks))
for x_subtick, x_subtick_label in zip(x_subticks, x_subtick_labels):
  ax0.text(x_subtick, ymin - (ymax - ymin) * 0.04, x_subtick_label, ha='center', va='top', fontsize=15)
for x_tick, x_tick_label in zip(np.array([horiz_major_tick * x_tick_idx for x_tick_idx in range(len(dim2))]), dim2):
  ax0.text(x_tick, ymin - (ymax - ymin) * 0.15, x_tick_label, ha='center', va='top', fontsize=20)
ax0.tick_params(which='major', width=1.6, length=9)
ax0.set_xlim([-horiz_margin * horiz_major_tick, (len(dim2) - 1 + horiz_margin) * horiz_major_tick])
if ytick is not None:
  yticks = np.arange(ymin, ymax + 1, ytick)
  ax0.set_yticks(yticks)
  ax0.set_yticklabels([str(y) for y in yticks])
ax0.set_ylim([ymin, ymax])
if normalize_base is None or len(normalize_base) == 0:
  if unit is None or len(unit) == 0:
    ax0.set_ylabel(f"{ylabel}", fontsize=20)
  else:
    ax0.set_ylabel(f"{ylabel} ({unit})", fontsize=20)
else:
  for x_tick, data in zip(x_tick_array[:, normalize_base_idx], agg_slice[:, normalize_base_idx]):
    ax0.text(x_tick + bar_width * 0.1, 1 + 0.02 * (ymax - ymin), f"{data:.2f} {unit}", ha="center", va="bottom", rotation=90, fontsize=14)
  ax0.set_ylabel(f"{ylabel}", fontsize=20)
ax0.yaxis.grid(zorder=0)

if force_ylim:
  ax0.set_ylim([ymin, ymax])
ymin, ymax = ax0.get_ylim()

for workload_idx, workload in enumerate(dim2):
  if normalize_base is None or len(normalize_base) == 0:
    normalize_divisor = 1
  else:
    normalize_divisor = normalize_base_count[workload_idx]
  to_plot_array = agg_slice[workload_idx, :] / normalize_divisor
  for setting_idx, to_plot_bar in enumerate(to_plot_array):
    if to_plot_bar > ymax:
      # ax0.text(x_tick_array[workload_idx, setting_idx] + bar_width * 0.1, ymax * 1.02, f"{to_plot_bar:.2f}x", ha="center", va="bottom", fontsize=13, rotation=90)
      ax0.text(x_tick_array[workload_idx, setting_idx] + bar_width * 0.1, ymax * 0.98, f"{to_plot_bar:.2f}x", ha="center", va="top", fontsize=13, rotation=90, color="white")

handles, labels = ax0.get_legend_handles_labels()
handles, labels = handles[0:len(handles):len(dim2)], labels[0:len(labels):len(dim2)]
legend = ax0.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.45, 1.4), ncol=len(labels))
ax0.hlines(0, xmin=ax0.get_xlim()[0], xmax=ax0.get_xlim()[1], zorder=9, color='black', linewidth=1)

pdf = PdfPages(fig_file)
pdf.savefig(bbox_inches="tight")
pdf.close()

def plot_stacked_bar():
  pass