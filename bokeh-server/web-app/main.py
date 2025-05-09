# NinjaNIRS 2022 - Raspberry Pi Zero SPI byte stream recorder
#
# bzim@bu.edu
# Initial version 2023-9-1
#
# start with : bokeh serve nn_bokeh_server4.py --allow-websocket-origin=ninja-pi.local:5006

import time
from threading import Thread
from functools import partial
import struct
import numpy as np
import socket
import datetime
import subprocess
from scipy.io import loadmat
from scipy.signal import welch
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from bokeh.layouts import column, row, Spacer, layout
from bokeh.models import Button, ColumnDataSource, TextInput,  Div, Range1d, DataRange1d, RadioButtonGroup, CheckboxGroup, ToggleButtonGroup, CheckboxButtonGroup
from bokeh.plotting import figure, curdoc
from bokeh.io import curdoc as curdoc2
from bokeh.models import CustomJS
from bokeh.palettes import Viridis256
import sys
sys.path.append('../data-server')
import nn_shared_memory

import probe_plot, plot_probe_power_calib

print("\n-----  NinjaNIRS 2024 byte bokeh server -- using NN24SystemClass -----\n")

# ADD THIS LINE here, after doc = curdoc()

# create a plot and style its properties
# source = ColumnDataSource(data=dict(x=[0], y=[0]))
# p = figure(width=1000, height=400,  y_range=(0, 1), x_range=Range1d(0, 30))
# p.xaxis.axis_label = "Time [s]"
# p.yaxis.axis_label = "Voltage [V]"
#
# l = p.line(x='x', y='y', source=source, line_width=2)

initial_time_window = 10
buffer_size = initial_time_window * 10
source = ColumnDataSource(data=dict(x=np.zeros(buffer_size), y=np.zeros(buffer_size)))
# p = figure(width=1000, height=400, y_range=DataRange1d(), x_range=Range1d(0, 10)) # Initial x-range
p = figure(width=1000, height=400, x_range=Range1d(0, 10)) # Initial x-range
p.xaxis.axis_label = "Time [s]"
p.yaxis.axis_label = "Voltage [V]"
l = p.line(x='x', y='y', source=source, line_width=2)

doc = curdoc()

# connect to shared memory
sm = nn_shared_memory.NNSharedMemory(main=False)

# Variables to track previous state for transition detection in blocking_task
prev_power_calib_status = False
current_time_window = initial_time_window
async def update(x, y):
    # source.stream(dict(x=x, y=y), rollover=300)

    new_data = dict(x=x, y=y)
    source.stream(new_data, rollover=buffer_size)

    # 2. Get the current x and y data from the source
    all_x = np.array(source.data['x'])
    all_y = np.array(source.data['y'])

    # Update the x-range to show the latest time window
    latest_time = x[-1] if x else 0
    p.x_range.start = latest_time - current_time_window  # Show a 30-second window
    p.x_range.end = latest_time

    # visible_mask = (all_x >= p.x_range.start) & (all_x <= p.x_range.end)
    # visible_y = all_y[visible_mask]
    #
    # y_min = np.min(visible_y)
    # y_max = np.max(visible_y)
    # y_padding = (y_max - y_min) * 0.1
    # if y_padding == 0:  # Handle case where all visible y values are the same
    #     y_padding = 0.1  # Or some small fixed value
    # p.y_range.start = y_min - y_padding
    # p.y_range.end = y_max + y_padding

# --- UI Update function triggered by server state changes ---
def update_darkplots_and_ui():
    """Updates dark plots and UI when power calibration finishes on the server."""

    # Update dark state plots
    power_calib_status_div.text = "<span style='color: green;'>Plotting dark state results</span>"
    # load dark states data
    file_path = os.path.join('..', 'meas', 'tmp_runtime_data','dark.mat')
    dark_mat =loadmat(file_path)
    data_dark = dark_mat['data_dark'].flatten()

    # --- Clear previous plot elements ---
    power_calib_dark_level.renderers = []
    power_calib_dark_level.xaxis.visible = False
    power_calib_dark_level.yaxis.visible = False
    power_calib_dark_level.xgrid.grid_line_color = None
    power_calib_dark_level.ygrid.grid_line_color = None
    power_calib_dark_level.background_fill_color = "white"  # Match original background
    power_calib_dark_level.title.text = "Dark Signal Levels"  # Add a title

    # --- Plot Sources (Red Circles) ---
    source_cds = ColumnDataSource(data=dict(x=src_x, y=src_y))
    power_calib_dark_level.scatter(x='x', y='y', source=source_cds, marker='circle', size=5, color='red',
                                   line_color=None)

    # --- Plot Detectors (Blue Circles) ---
    detector_cds = ColumnDataSource(data=dict(x=det_x, y=det_y))
    power_calib_dark_level.scatter(x='x', y='y', source=detector_cds, marker='circle', size=5, color='blue',
                                   line_color=None)

    # --- Prepare data for Dark Signal Lines (Edges) ---
    edge_xs = []
    edge_ys = []
    edge_colors = []
    cmap_func = cm.get_cmap('jet')
    colors_list = cmap_func(np.linspace(0, 1, 64))

    lst = np.where(ml[:, 3] == 1)[0]

    for i in lst:
        s_idx = int(ml[i, 0]) - 1  # Source index (1-based -> 0-based)
        d_idx = int(ml[i, 1]) - 1  # Detector index (1-based -> 0-based)
        dDark_val = data_dark[i]
        xcoords = [src_x[s_idx], det_x[d_idx]]
        ycoords = [src_y[s_idx], det_y[d_idx]]

        log_val = 10 * np.log10(np.abs(dDark_val))
        ceil_val = np.ceil(log_val)
        clamped_val = np.maximum(-63.0, np.minimum(ceil_val, 0.0))
        icm_float = clamped_val + 64.0
        icm = int(icm_float)
        idx = max(0, min(icm - 1, 63)) # Index is now 1-based integer [0, 63]
        rgb_color = colors_list[idx]
        color_hex = mcolors.to_hex(rgb_color, keep_alpha=False)
        edge_xs.append(xcoords)
        edge_ys.append(ycoords)
        edge_colors.append(color_hex)

    edge_source = ColumnDataSource(data=dict(xs=edge_xs, ys=edge_ys, color=edge_colors))
    power_calib_dark_level.multi_line(xs='xs', ys='ys', color='color', source=edge_source, line_width=1.5)

    p_psd = power_calib_dark_level_ps
    data_states_dark = dark_mat['data_states_dark']
    B_permuted = np.transpose(data_states_dark, (2, 0, 1))
    B = B_permuted.reshape((B_permuted.shape[0] * B_permuted.shape[1], B_permuted.shape[2]))
    fs = 800.0  # Sampling frequency - VERIFY
    ysum = None
    f = None
    n = 0
    for ii in range(B.shape[1]):
        d = NN22_interpNAN(B[:, ii])
        if 1:  # np.mean(d) < 1e6:
            f, y = welch(d, fs=800)
            if ysum is None:
                ysum = np.sqrt(y)
            else:
                ysum += np.sqrt(y)
            n += 1
    ysum /= n
    p_psd.renderers = []
    p_psd.xaxis.visible = True
    p_psd.yaxis.visible = True
    p_psd.xgrid.grid_line_color = 'lightgray'
    p_psd.ygrid.grid_line_color = 'lightgray'
    p_psd.xaxis.axis_label = "Frequency (Hz)"
    p_psd.title.text = 'Power Spectrum Of Dark Signal'

    # Add the line data
    psd_source = ColumnDataSource(data=dict(f=f, y=ysum))
    p_psd.line(x='f', y='y', source=psd_source, line_width=2, color="navy")

    button_run_power_calib.label = "Power Calibration"
    button_run_power_calib.button_type = "default"

    current_sig_level_tuning_status = sm.getStatus('sig_level_tuning')

    if current_sig_level_tuning_status:
        # If server automatically moved to signal tuning, highlight that button
        button_signal_level.button_type = "success"
        power_calib_status_div.text = "<span style='color: green;'>Power Calibration Finished. Signal Level Adjustment active.</span>"
    else:
        # If server did NOT automatically move to signal tuning, just set signal level button to default
        button_signal_level.button_type = "default"
        power_calib_status_div.text = "<span style='color: green;'>Power Calibration finished</span>"

def NN22_interpNAN(d):
    d = np.array(d)

    lst1 = np.where(np.isnan(d[1:]) & ~np.isnan(d[:-1]))[0] + 1
    lst2 = np.where(np.isnan(d[:-1]) & ~np.isnan(d[1:]))[0]

    while len(lst1) > 0 or len(lst2) > 0:
        d[lst1] = d[lst1 - 1]
        d[lst2] = d[lst2 + 1]

        lst1 = np.where(np.isnan(d[1:]) & ~np.isnan(d[:-1]))[0] + 1
        lst2 = np.where(np.isnan(d[:-1]) & ~np.isnan(d[1:]))[0]

    return d


def button_run_callback():
    sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = True
    source.data = dict(x=[], y=[])
    print("Run button pressed")
    # p.renderers = []

def button_stop_callback():
    sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = False
    print("Stop button pressed")
    # Reset shared memory disp time values
    for i in range(len(sm.disp_rbuf_time)):
        sm.disp_rbuf_time[i] = np.nan

def button_enable_power_calib_callback():
    button_signal_level.disabled = False
    button_run_power_calib.disabled = False
    button_return.disabled = False
    power_calib_status_div.text = "<span style='color: red;'>Press power calibration button to start power calibration</span>"  # Change color when stopped

# --- Callbacks for the three individual mode buttons ---
def button_signal_level_callback():
    if not sm.status_shm.buf[sm.STATUS_SHM_IDX['power_calib']]:
        sm.status_shm.buf[sm.STATUS_SHM_IDX['sig_level_tuning']] = True
        sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = True
        button_signal_level.button_type = "success"
        power_calib_status_div.text = "<span style='color: green;'>Signal Level Adjustment active.</span>"
    # sm.status_shm.buf[sm.STATUS_SHM_IDX['sig_level_tuning']] = True
    # sm.status_shm.buf[sm.STATUS_SHM_IDX['power_calib']] = False
    # sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = False # Activating this mode stops others
    # print("Mode: Signal level adjustment")

def button_run_power_calib_callback():
    if not sm.status_shm.buf[sm.STATUS_SHM_IDX['power_calib']]:
        sm.status_shm.buf[sm.STATUS_SHM_IDX['sig_level_tuning']] = False
        sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = False
        sm.status_shm.buf[sm.STATUS_SHM_IDX['power_calib']] = True
        button_run_power_calib.label = "Calibrating..."
        button_run_power_calib.button_type = "success"
        button_signal_level.button_type = "default"
        power_calib_status_div.text = "<span style='color: green;'>Acquiring power calibration data. It will take less than 30 seconds</span>"
    # sm.status_shm.buf[sm.STATUS_SHM_IDX['power_calib']] = True
    # sm.status_shm.buf[sm.STATUS_SHM_IDX['sig_level_tuning']] = False
    # sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = False # Activating this mode stops others
    # print("Mode: Power Calibration")

def button_return_callback():
    if not sm.status_shm.buf[sm.STATUS_SHM_IDX['power_calib']]:
        sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = False
        sm.status_shm.buf[sm.STATUS_SHM_IDX['sig_level_tuning']] = False
        button_run_power_calib.label = "Power Calibration"
        button_run_power_calib.button_type = "default"
        button_signal_level.button_type = "default"
        power_calib_status_div.text = "<span style='color: green;'></span>"
        button_signal_level.disabled = True
        button_run_power_calib.disabled = True
        button_return.disabled = True
    # sm.status_shm.buf[sm.STATUS_SHM_IDX['sig_level_tuning']] = False
    # sm.status_shm.buf[sm.STATUS_SHM_IDX['power_calib']] = False
    # sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = False # Activating this mode stops others
    # print("Mode: Return (all calibration/run off)")

def spatial_multiplexing_callback(attr, old, new):
    pass
    # Update shared memory based on the checkbox state
    # is_active = 0 in new # Checkboxes return a list of active indices
    # sm.status_shm.buf[sm.STATUS_SHM_IDX['spatial_multiplexing_enabled']] = is_active
    # print(f"Spatial Multiplexing: {'Enabled' if is_active else 'Disabled'}")

def fast_calib_callback(attr, old, new):
    pass
    # Update shared memory based on the checkbox state
    # is_active = 0 in new
    # sm.status_shm.buf[sm.STATUS_SHM_IDX['fast_calib_enabled']] = is_active
    # print(f"Fast Power Calibration: {'Enabled' if is_active else 'Disabled'}")

# def sds_range_callback(attr, old, new):
#     print(f"SDS Range updated: {new}")
#     # sm.set_config_string('sds_range', new) # Need to implement this in nn_shared_memory

# Callback for Wavelength RadioButtonGroup
def wavelength_callback(attr, old, new):
    selected_wavelength = new + 1
    sm.plot_wavelength.buf[:2] = struct.pack('H', selected_wavelength)

# Callback for Delta OD CheckboxGroup
def delta_od_callback(attr, old, new):
    if new:
        sm.status_shm.buf[sm.STATUS_SHM_IDX['delta_OD']] = True
    else:
        sm.status_shm.buf[sm.STATUS_SHM_IDX['delta_OD']] = False

def time_window_callback(attr, old, new):
    global current_time_window
    try:
        new_window = float(new)
        if new_window > 0:
            current_time_window = new_window
            # Update plot range based on the latest data point and new window
            if len(source.data['x']) > 0:
                latest_time = source.data['x'][-1]
                p.x_range.start = latest_time - current_time_window
                p.x_range.end = latest_time
        else:
            time_window_input.value = str(current_time_window) # Revert to previous valid value
    except ValueError:
        time_window_input.value = str(current_time_window) # Revert to previous valid value

def get_disk_space():
    try:
        # Execute the df -h / command to get human-readable disk usage for the root filesystem
        result = subprocess.run(['df', '-h', '/'], capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        # The line with the data is the second line
        if len(lines) > 1:
            parts = lines[1].split()
            # Available space is the 4th column (index 3)
            available = parts[3]
            return f"Available SD Card Space: {available}"
        return "Could not parse disk space."
    except FileNotFoundError:
        return "Command 'df' not found."
    except subprocess.CalledProcessError as e:
        return f"Error getting disk space: {e}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

# Callback for the update memory button
def update_memory_callback():
    memory_div.text = get_disk_space()

def blocking_task():
    global prev_power_calib_status
    x0=0
    while True:
        current_power_calib_status = sm.getStatus('power_calib')
        if prev_power_calib_status == True and current_power_calib_status == False:
            doc.add_next_tick_callback(partial(update_darkplots_and_ui))
        if sm.getStatus('disp_rbuf_wr_idx') != sm.getStatus('disp_rbuf_rd_idx'):
            if not sm.getStatus('sig_level_tuning'):
                y = sm.disp_rbuf[sm.getStatus('disp_rbuf_rd_idx')%sm.DISP_RBUF_SIZE]
                x = sm.disp_rbuf_time[sm.getStatus('disp_rbuf_rd_idx')%sm.DISP_RBUF_SIZE]
                sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_rd_idx']] = (sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_rd_idx']] + 1) % sm.DISP_RBUF_SIZE
                # x = x0/800
                doc.add_next_tick_callback(partial(update, x=[x], y=[y]))
                x0 += 1
            else:
                # get ml_sig_vales
                shared_arr_ml_sig = np.ndarray(ml_length, dtype=np.int16, buffer=sm.ml_sig_values.buf)
                # ml_sig_values = np.copy(shared_arr_ml_sig)
                # shared_arr_ = np.ndarray(ml_length, dtype=np.int16, buffer=sm.ml_sig_values.buf)
                ml_sig_values = np.copy(shared_arr_ml_sig)
                shared_arr_n_poor_srcs = np.ndarray(n_srcs, dtype=np.int16, buffer=sm.n_poor_srcs.buf)
                n_poor_srcs = np.copy(shared_arr_n_poor_srcs)
                # print('n_poor_srcs ~ ', n_poor_srcs)
                shared_arr_n_poor_dets = np.ndarray(n_dets, dtype=np.int16, buffer=sm.n_poor_dets.buf)
                n_poor_dets = np.copy(shared_arr_n_poor_dets)
                # print('n_poor_dets ~ ', n_poor_dets)
                doc.add_next_tick_callback(partial(update_power_calib_plot_safe, power_calib_probe_panel, power_calib_graph.edge_renderer.data_source, power_calib_label_source, power_calib_node_source, ml_sig_values,
                                                   n_poor_srcs, n_poor_dets, poor_snr_srcs_cds, poor_snr_dets_cds))
                time.sleep(0.5) # this gives time to update calibration part dynamically
        else:
            time.sleep(0.001)

        prev_power_calib_status = current_power_calib_status

def update_power_calib_plot(plot, edge_data_source, ml_sig_values):
    """
    Dynamically updates the visibility and color of edges based on ml_sig_values.

    Args:
        plot (bokeh.plotting.figure): The Bokeh plot object.
        edge_data_source (bokeh.models.ColumnDataSource): The data source for the edges.
        ml_sig_values (numpy.ndarray): A 1D numpy array of signal values for each edge.
    """
    num_edges = len(edge_data_source.data['start'])
    if len(ml_sig_values) != num_edges:
        print(f"Error: Length of ml_sig_values ({len(ml_sig_values)}) does not match the number of edges ({num_edges}).")
        return

    new_alpha = []
    new_line_color = []
    colormap = Viridis256 # Using the Viridis256 palette

    for i, sig_value in enumerate(ml_sig_values):
        if sig_value == 1:
            new_alpha.append(0)
            new_line_color.append("red") # Keep a default color when hidden
        elif sig_value == 2:
            new_alpha.append(1)
            new_line_color.append("red") # Show with a default color
        elif sig_value <= 0:
            color_index = int(abs(sig_value))
            # Adjust index to be within the colormap range (0 to 255)
            adjusted_index = color_index - 1
            if 0 <= adjusted_index < len(colormap):
                new_alpha.append(1)
                new_line_color.append(colormap[adjusted_index])
            else:
                new_alpha.append(1)
                new_line_color.append("red") # Default color if index is out of range
        else:
            # Handle other cases if needed, for now, show with default
            new_alpha.append(1)
            new_line_color.append("slategrey")

    edge_data_source.data['alpha'] = new_alpha
    edge_data_source.data['line_color'] = new_line_color

    # Trigger an update of the plot
    edge_data_source.trigger('data', edge_data_source.data, edge_data_source.data)


def update_power_calib_plot_safe(plot, edge_data_source, label_data_source, node_data_source,  ml_sig_values, n_poor_srcs, n_poor_dets, poor_snr_srcs_cds, poor_snr_dets_cds):
    """
    Safely updates the visibility and color of edges based on ml_sig_values
    within the Bokeh server's event loop.
    """
    n_src = len(n_poor_srcs)
    n_det = len(n_poor_dets)
    num_nodes = n_src + n_det
    num_edges = len(edge_data_source.data['start'])
    if len(ml_sig_values) != num_edges:
        print(f"Error: Length of ml_sig_values ({len(ml_sig_values)}) does not match the number of edges ({num_edges}).")
        return

    new_alpha = []
    new_line_color = []
    colormap = Viridis256 # Using the Viridis256 palette
    print(ml_sig_values)

    for i, sig_value in enumerate(ml_sig_values):
        if sig_value == 1:
            new_alpha.append(0)
            new_line_color.append("red") # Keep a default color when hidden
        elif sig_value == 2:
            new_alpha.append(1)
            new_line_color.append("red") # Show with a default color
        elif sig_value <= 0:
            color_index = int(abs(sig_value))
            # Adjust index to be within the colormap range (0 to 255)
            adjusted_index = color_index - 1
            new_alpha.append(1)
            new_line_color.append("blue")
            # if 0 <= adjusted_index < len(colormap):
            #     new_alpha.append(1)
            #     new_line_color.append(colormap[adjusted_index])
            # else:
            #     new_alpha.append(1)
            #     new_line_color.append("green") # Default color if index is out of range
        else:
            # Handle other cases if needed, for now, show with default
            new_alpha.append(1)
            new_line_color.append("green")

    edge_data_source.data = dict(alpha=new_alpha, line_color=new_line_color, start=edge_data_source.data['start'], end=edge_data_source.data['end'])

    # update top 6 poor srcs/dets as labels and rest as nodes
    order_poor_srcs = np.argsort(n_poor_srcs)[::-1]
    order_poor_dets = np.argsort(n_poor_dets)[::-1]
    top_poor_src_indices = {idx for idx in order_poor_srcs[:6] if n_poor_srcs[idx] > 0}
    top_poor_det_indices = {idx for idx in order_poor_dets[:6] if n_poor_dets[idx] > 0}
    new_node_alpha = [1.0] * num_nodes
    new_label_alpha = [0.0] * num_nodes

    # Process top poor sources: Hide circle, show label
    for src_idx in top_poor_src_indices:
        new_node_alpha[src_idx] = 0.0  # Hide circle for top poor source
        new_label_alpha[src_idx] = 1.0  # Show label for top poor source

    # Process top poor detectors: Hide circle, show label
    for det_idx in top_poor_det_indices:
        node_idx = n_src + det_idx  # Calculate the overall node index
        new_node_alpha[node_idx] = 0.0  # Hide circle for top poor detector
        new_label_alpha[node_idx] = 1.0  # Show label for top poor detector

    node_data_source.data['alpha'] = new_node_alpha
    label_data_source.data['alpha'] = new_label_alpha

    # update nodes numbers in the label box
    update_poor_snr_figures(n_poor_srcs, n_poor_dets, poor_snr_srcs_cds, poor_snr_dets_cds)

def update_poor_snr_figures(n_poor_srcs, n_poor_dets, src_cds, det_cds):
    # For Sources
    # print('poor_srcs_shape',n_poor_srcs.shape)
    # print('poor_dets_shape', n_poor_dets.shape)
    order_poor_srcs = np.argsort(n_poor_srcs)[::-1] # Sort in descending order and get indices
    src_str = ''
    for i in range(min(10, len(order_poor_srcs))):
        index = order_poor_srcs[i]
        if n_poor_srcs[index] > 0:
            src_str += f"S{index + 1} ({n_poor_srcs[index]})\n"
            # src_str += f"S{index + 1} \n"

    # For Detectors
    order_poor_dets = np.argsort(n_poor_dets)[::-1] # Sort in descending order and get indices
    det_str = ''
    for i in range(min(10, len(order_poor_dets))):
        index = order_poor_dets[i]
        if n_poor_dets[index] > 0:
            det_str += f"D{index + 1} ({n_poor_dets[index]})\n"

    src_cds.data['text'] = [src_str]
    det_cds.data['text'] = [det_str]

def is_connected_to_internet(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False

# Python callback when button is clicked
def on_button_click():
    client_time = time_input.value
    if client_time:
        print( f"Button pressed at (client date & time): {client_time}")
        # output.text = f"Button pressed at (client date & time): {client_time}"
    else:
        print("No client time captured. Try again.")
        # output.text = "No client time captured. Try again."

def on_time_update(attr, old, date_str):
    print( f"Button pressed at (client date & time): {date_str}")
    dt_obj = datetime.datetime.strptime(date_str, "%m/%d/%Y, %I:%M:%S %p")
    formatted_date = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Button pressed at (client date & time): {formatted_date}")
    try:
        # Format: 'YYYY-MM-DD HH:MM:SS'
        subprocess.run(['sudo', 'date', '-s', formatted_date], check=True)
        print("System time updated successfully.")
        button_set_date_time.label = "Date and Time was set"
        button_set_date_time.button_type = 'default'
        button_set_date_time.disabled = True
    except subprocess.CalledProcessError as e:
        print(f"Failed to set system time: {e}")
        button_set_date_time.label = "Error! Try again"

if is_connected_to_internet():
    button_set_date_time = Button(label="Date and Time was set", align="center", width=200, disabled=True)
else:
    time_input = TextInput(value="", visible=False)
    button_set_date_time = Button(label="Set date and time", align="center", width=200, button_type='warning')

    # JavaScript to get client’s local date & time
    js_code = """
            var now = new Date();
            var timestamp = now.toLocaleString();  // Gets date & time as a string
            time_input.value = timestamp;  // Store it in hidden TextInput

            console.log("JavaScript executed, new time: " + timestamp);
        """
    button_set_date_time.js_on_click(CustomJS(args=dict(time_input=time_input), code=js_code))
    time_input.on_change('value', on_time_update)


# add a button widget and configure with the call back
button_run = Button(label='Run', width=200, align="center")
button_run.on_event('button_click', button_run_callback)

# add a button widget and configure with the call back
button_stop = Button(label='Stop', width=200, align = "center")
button_stop.on_event('button_click', button_stop_callback)

# add a button widget and configure with the call back
button_enable_power_calib = Button(label='Enable Power Calib', width=200, align = "center")
button_enable_power_calib.on_event('button_click', button_enable_power_calib_callback)

memory_div = Div(text=get_disk_space(), width=200, align='center')
update_memory_button = Button(label="Refresh Memory", width=200, align='center')
update_memory_button.on_event('button_click', update_memory_callback)



# Add CheckboxGroup for Delta OD
delta_od_checkbox = CheckboxGroup(
    labels=["Delta OD"], active=[], width=200, align="center"
)
delta_od_checkbox.on_change('active', delta_od_callback)

# Add TextInput for Time Window Length
time_window_label = Div(text="Time Window [s]:", width=100, align="end")
time_window_input = TextInput(value=str(initial_time_window), width=100)
time_window_input.on_change('value', time_window_callback)

time_window_row = row(time_window_label, time_window_input, width=200, align="center")


power_calib_status_div = Div(text="<span style='color: red;'> </span>", width=250, align='center')

# wavelength_delta_od_row = row(
#     wavelength_radio_button_group,
#     delta_od_checkbox,
#     align="center", # Align the row horizontally
#     width=200 # Give the row a combined width
# )

probe_plot_panel, probe_graph, ml_length, sd_data = probe_plot.gen_probe_plot()
n_srcs = len(sd_data['SD']['SrcPos2D'][0][0])

# this below data will be useful for dark signal plots in power calib window
ml = sd_data['SD']['MeasList'][0][0]
src_pos = sd_data['SD']['SrcPos2D'][0][0]
det_pos = sd_data['SD']['DetPos2D'][0][0]
src_x = src_pos[:, 0]
src_y = src_pos[:, 1]
det_x = det_pos[:, 0]
det_y = det_pos[:, 1]

# Add RadioButtonGroup for Wavelength selection
wavelength_radio_button_group = RadioButtonGroup(labels=[str(sd_data['SD']['Lambda'][0][0][0][0])+'nm', str(sd_data['SD']['Lambda'][0][0][0][1])+'nm'], active=0, width=200, align="center")
wavelength_radio_button_group.on_change('active', wavelength_callback)

n_dets = len(sd_data['SD']['DetPos2D'][0][0])
# def select_callback(attr, old, new):
#     print('node selected')
#     inds = new
#     p.yaxis.axis_label = str(inds)
#     print(f"Selected indices {inds}")

def edge_select_callback(attr, old, inds):
    if inds:
        sm.plot_ml_idx.buf[:2] = struct.pack('H', inds[0])
    p.yaxis.axis_label = str(inds)
    print(f"Selected indices {inds}")

def gen_poor_srcs_dets_axis():

    # Data sources for poor SNR figures
    power_calib_poor_snr_srcs_cds = ColumnDataSource(data=dict(text=[])) # Changed 'indices' to 'text'
    power_calib_poor_snr_dets_cds = ColumnDataSource(data=dict(text=[])) # Changed 'indices' to 'text'

    # Poor SNR figures
    power_calib_poor_snr_srcs = figure(width=250, height=250, title="Poor SNR Sources", y_axis_label="",
                                       y_range=Range1d(0, 10, bounds=(0, 10))) # Removed y-axis label
    power_calib_poor_snr_dets = figure(width=250, height=250, title="Poor SNR Detectors", y_axis_label="",
                                       y_range=Range1d(0, 10, bounds=(0, 10))) # Removed y-axis label

    # Add text glyphs to display the string
    power_calib_poor_snr_srcs.text(x=0.05, y=9.5, text='text', source=power_calib_poor_snr_srcs_cds,
                                   text_align='left', text_baseline='top', text_color='red') # Adjusted y and baseline
    power_calib_poor_snr_dets.text(x=0.05, y=9.5, text='text', source=power_calib_poor_snr_dets_cds,
                                   text_align='left', text_baseline='top', text_color='blue') # Adjusted y and baseline
    power_calib_poor_snr_srcs.xgrid.grid_line_color = None
    power_calib_poor_snr_srcs.xaxis.visible = False
    power_calib_poor_snr_srcs.yaxis.visible = False # Hide y-axis
    power_calib_poor_snr_dets.xgrid.grid_line_color = None
    power_calib_poor_snr_dets.xaxis.visible = False
    power_calib_poor_snr_dets.yaxis.visible = False # Hide y-axis

    return power_calib_poor_snr_srcs, power_calib_poor_snr_dets, power_calib_poor_snr_srcs_cds, power_calib_poor_snr_dets_cds


# probe_graph.node_renderer.data_source.selected.on_change('indices', select_callback)
probe_graph.edge_renderer.data_source.selected.on_change('indices', edge_select_callback)

# put the buttons and plot in a layout and add to the document

# button_panel = column(Spacer(height=100), button_set_date_time, button_run, button_stop,button_run_power_calib, button_stop_power_calib, Spacer(width=500))
button_panel = column(
    Spacer(height=100),
    button_set_date_time,
    update_memory_button,
    memory_div,
    button_run,
    button_stop,
    button_enable_power_calib,
    Spacer(height=20), # Add some space
    wavelength_radio_button_group,
    delta_od_checkbox,
    time_window_row,
    Spacer(width=500) # Keep the original spacer for alignment
)
button_signal_level = Button(label="Signal level adjustment", width=200, align='center', disabled=True)
button_run_power_calib = Button(label="Power Calibration", width=200, align='center', disabled=True)
button_return = Button(label="Return", width=200, align='center', disabled=True)

button_signal_level.on_event('button_click', button_signal_level_callback)
button_run_power_calib.on_event('button_click', button_run_power_calib_callback)
button_return.on_event('button_click', button_return_callback)

spacer_top = Spacer(sizing_mode='stretch_height')
spacer_bottom = Spacer(sizing_mode='stretch_height')

# Accelerometer Status
if sm.SYS_STATUS.buf[5]:
    accel_status_div = Div(text="<span style='color: green;'>Accelerometer Status: Connected </span>", width=200, align='center') # Initial status, will be updated
else:
    accel_status_div = Div(text="<span style='color: red;'>Accelerometer Status: Not Connected </span>", width=200, align='center') # Initial status, will be updated

# Checkboxes for Calibration Options
spatial_multiplexing_checkbox = CheckboxGroup(labels=["Spatial Multiplexing"], active=[0], width=200, align='center') # active=[] means unchecked
fast_calib_checkbox = CheckboxGroup(labels=["Fast Power Calibration"], active=[], width=200, align='center') # active=[] means unchecked

# Attach callbacks to checkboxes
spatial_multiplexing_checkbox.on_change('active', spatial_multiplexing_callback)
fast_calib_checkbox.on_change('active', fast_calib_callback)


# SDS Range Input
# sds_range_label = Div(text="SDS range:", width=100, align='end') # Align label to end of its Div
# sds_range_input = TextInput(value="3-15", width=100) # Default value
#
# sds_range_row = row(
#     sds_range_label,
#     sds_range_input,
#     width=250, # Give the row a width
#     align='center' # Center the label and input horizontally within the row
# )
#
# sds_range_input.on_change('value', sds_range_callback)




# Calibration panel label_data_source, node_data_source
power_calib_probe_panel, power_calib_graph, power_calib_labels, power_calib_label_source, power_calib_node_source, power_calib_edge_source = plot_probe_power_calib.gen_probe_plot(sd_data)
# power_calib_button_panel1 = figure( width=250, height=250)
power_calib_button_panel1 = column(spacer_top,
                                   button_signal_level,
                                   button_run_power_calib,
                                   button_return,
                                   Spacer(height=20),
                                   power_calib_status_div,
                                   spacer_bottom, width=250, height=250, align='center')
# power_calib_button_panel2 = figure( width=250, height=250)
power_calib_button_panel2 = column(
    Spacer(sizing_mode='stretch_height'), # Spacer for vertical centering
    accel_status_div,
    spatial_multiplexing_checkbox,
    fast_calib_checkbox,
    # sds_range_row,
    Spacer(sizing_mode='stretch_height'), # Spacer for vertical centering
    width=250,
    height=250, # Explicit height for this panel container
    align='center' # Center widgets horizontally within this column
)
# power_calib_poor_snr_srcs = figure( width=250, height=250)
# power_calib_poor_snr_dets = figure( width=250, height=250)
power_calib_poor_snr_srcs, power_calib_poor_snr_dets, poor_snr_srcs_cds, poor_snr_dets_cds = gen_poor_srcs_dets_axis()
power_calib_sig_sds = figure( width=333, height=333)
power_calib_dark_level = figure( width=333, height=333)
power_calib_dark_level_ps = figure( width=333, height=333, y_axis_type="log")

power_calib_button_panel = row(power_calib_button_panel1, power_calib_button_panel2)
power_calib_poor_snr_optodes = row(power_calib_poor_snr_srcs, power_calib_poor_snr_dets)
power_calib_bottom_panel = row(power_calib_sig_sds, power_calib_dark_level, power_calib_dark_level_ps)

power_calib_right_top_panel = column(power_calib_button_panel, power_calib_poor_snr_optodes)
power_calib_top_panel = row(power_calib_probe_panel, power_calib_right_top_panel)

power_calib_panel = column(power_calib_top_panel, power_calib_bottom_panel)

grid_panel = layout(
        children=[
            [button_panel, probe_plot_panel],
            [p],
            [power_calib_panel]
        ],
        align='center'
        #sizing_mode='fixed',
    )

# Create an outer layout to center the grid_panel
outer_layout = column(
    grid_panel,
    sizing_mode='stretch_width', # Make this column fill the width
    align='center' # Center the grid_panel within this column
)

# doc.add_root(grid_panel)
doc.add_root(outer_layout)

# doc.add_root(column(row(column(button_run, button_stop, align='center'), probe_plot_panel, align='center'), p, align='center'))
thread = Thread(target=blocking_task)
thread.start()

