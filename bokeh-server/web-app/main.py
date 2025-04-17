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

from bokeh.layouts import column, row, Spacer, layout
from bokeh.models import Button, ColumnDataSource, TextInput,  Div, Range1d
from bokeh.plotting import figure, curdoc
from bokeh.io import curdoc as curdoc2
from bokeh.models import CustomJS
from bokeh.palettes import Viridis256
import sys
sys.path.append('../data-server')
import nn_shared_memory

import probe_plot, plot_probe_power_calib

print("\n-----  NinjaNIRS 2024 byte bokeh server -- using NN24SystemClass -----\n")

# create a plot and style its properties
source = ColumnDataSource(data=dict(x=[0], y=[0]))
p = figure(width=1000, height=400,  y_range=(0, 3))
p.xaxis.axis_label = "Time [s]"
p.yaxis.axis_label = "Voltage [V]"

l = p.line(x='x', y='y', source=source, line_width=2)

doc = curdoc()

# connect to shared memory
sm = nn_shared_memory.NNSharedMemory(main=False)

async def update(x, y):
    source.stream(dict(x=x, y=y), rollover=2500)

def button_run_callback():
    sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = True
    source.data = dict(x=[], y=[])
    print("Run button pressed")
    # p.renderers = []

def button_stop_callback():
    sm.status_shm.buf[sm.STATUS_SHM_IDX['run']] = False
    print("Stop button pressed")
    sm.status_shm.buf[sm.STATUS_SHM_IDX['sig_level_tuning']] = False
    # Reset shared memory disp time values
    for i in range(len(sm.disp_rbuf_time)):
        sm.disp_rbuf_time[i] = np.nan

def button_run_power_calib_callback():
    sm.status_shm.buf[sm.STATUS_SHM_IDX['power_calib']] = True

def button_stop_power_calib_callback():
    sm.status_shm.buf[sm.STATUS_SHM_IDX['power_calib']] = False


def blocking_task():
    x0=0
    while True:
        if sm.getStatus('disp_rbuf_wr_idx') != sm.getStatus('disp_rbuf_rd_idx'):
            if not sm.getStatus('sig_level_tuning'):
                y = sm.disp_rbuf[sm.getStatus('disp_rbuf_rd_idx')%sm.DISP_RBUF_SIZE]
                print(y)
                x = sm.disp_rbuf_time[sm.getStatus('disp_rbuf_rd_idx')%sm.DISP_RBUF_SIZE]
                sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_rd_idx']] = (sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_rd_idx']] + 1) % sm.DISP_RBUF_SIZE
                # x = x0/800
                doc.add_next_tick_callback(partial(update, x=[x], y=[y]))
                x0 += 1
            else:
                # get ml_sig_vales
                shared_arr_ml_sig = np.ndarray(ml_length, dtype=np.int16, buffer=sm.ml_sig_values.buf)
                # ml_sig_values = np.copy(shared_arr_ml_sig)
                # print(ml_sig_values )
                # shared_arr_ = np.ndarray(ml_length, dtype=np.int16, buffer=sm.ml_sig_values.buf)
                ml_sig_values = np.copy(shared_arr_ml_sig)
                shared_arr_n_poor_srcs = np.ndarray(n_srcs, dtype=np.int16, buffer=sm.n_poor_srcs.buf)
                n_poor_srcs = np.copy(shared_arr_n_poor_srcs)
                # print('n_poor_srcs ~ ', n_poor_srcs)
                shared_arr_n_poor_dets = np.ndarray(n_dets, dtype=np.int16, buffer=sm.n_poor_dets.buf)
                n_poor_dets = np.copy(shared_arr_n_poor_dets)
                # print('n_poor_dets ~ ', n_poor_dets)
                doc.add_next_tick_callback(partial(update_power_calib_plot_safe, power_calib_probe_panel, power_calib_graph.edge_renderer.data_source, ml_sig_values,
                                                   n_poor_srcs, n_poor_dets, poor_snr_srcs_cds, poor_snr_dets_cds))
                time.sleep(0.1) # this gives time to update calibration part dynamically
                # update_power_calib_plot(power_calib_probe_panel, power_calib_graph.edge_renderer.data_source, ml_sig_values)
        else:
            time.sleep(0.001)

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


def update_power_calib_plot_safe(plot, edge_data_source, ml_sig_values, n_poor_srcs, n_poor_dets, poor_snr_srcs_cds, poor_snr_dets_cds):
    """
    Safely updates the visibility and color of edges based on ml_sig_values
    within the Bokeh server's event loop.
    """
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
            new_line_color.append("blue")

    edge_data_source.data = dict(alpha=new_alpha, line_color=new_line_color, start=edge_data_source.data['start'], end=edge_data_source.data['end'])
    # print(f"edge_data_source.data after update: {edge_data_source.data}")  # Debugging print
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
            # src_str += f"S{index + 1} ({n_poor_srcs[index]})\n"
            src_str += f"S{index + 1} \n"

    # For Detectors
    order_poor_dets = np.argsort(n_poor_dets)[::-1] # Sort in descending order and get indices
    det_str = ''
    for i in range(min(10, len(order_poor_dets))):
        index = order_poor_dets[i]
        if n_poor_dets[index] > 0:
            det_str += f"D{index + 1}\n"

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
    except subprocess.CalledProcessError as e:
        print(f"Failed to set system time: {e}")

if is_connected_to_internet():
    print("Connected to the Internet.")
    button_set_date_time = Button(label="System connected internet", align="center")
else:
    print("No internet connection.")
    time_input = TextInput(value="", visible=False)
    button_set_date_time = Button(label="Set date and time", align="center")

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
button_run = Button(label='Run', align="center")
button_run.on_event('button_click', button_run_callback)

# add a button widget and configure with the call back
button_stop = Button(label='Stop', align = "center")
button_stop.on_event('button_click', button_stop_callback)

# add a button widget and configure with the call back
button_run_power_calib = Button(label='Run_Power_Calib', align = "center")
button_run_power_calib.on_event('button_click', button_run_power_calib_callback)

# add a button widget and configure with the call back
button_stop_power_calib = Button(label='Stop_Power_Calib', align = "center")
button_stop_power_calib.on_event('button_click', button_stop_power_calib_callback)

probe_plot_panel, probe_graph, ml_length, sd_data = probe_plot.gen_probe_plot()
n_srcs = len(sd_data['SD']['SrcPos2D'][0][0])
print('n_srcs:', n_srcs)
n_dets = len(sd_data['SD']['DetPos2D'][0][0])
print('n_dets:', n_dets)
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
                                   text_align='left', text_baseline='top', text_color='red') # Adjusted y and baseline
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

button_panel = column(Spacer(height=100), button_run, button_stop,button_run_power_calib, button_stop_power_calib, button_set_date_time, Spacer(width=500))

# Calibration panel
power_calib_probe_panel, power_calib_graph, power_calib_labels, power_calib_label_source = plot_probe_power_calib.gen_probe_plot(sd_data)
power_calib_button_panel1 = figure( width=250, height=250)
power_calib_button_panel2 = figure( width=250, height=250)
# power_calib_poor_snr_srcs = figure( width=250, height=250)
# power_calib_poor_snr_dets = figure( width=250, height=250)
power_calib_poor_snr_srcs, power_calib_poor_snr_dets, poor_snr_srcs_cds, poor_snr_dets_cds = gen_poor_srcs_dets_axis()
power_calib_sig_sds = figure( width=333, height=333)
power_calib_dark_level = figure( width=333, height=333)
power_calib_dark_level_ps = figure( width=333, height=333)

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

doc.add_root(grid_panel)

# doc.add_root(column(row(column(button_run, button_stop, align='center'), probe_plot_panel, align='center'), p, align='center'))
thread = Thread(target=blocking_task)
thread.start()

