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

from bokeh.layouts import column, row, Spacer, layout
from bokeh.models import Button, ColumnDataSource
from bokeh.plotting import figure, curdoc
from bokeh.models import CustomJS
import sys
sys.path.append('../data-server')
import nn_shared_memory

import probe_plot

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
            y = sm.disp_rbuf[sm.getStatus('disp_rbuf_rd_idx')%sm.DISP_RBUF_SIZE]
            x = sm.disp_rbuf_time[sm.getStatus('disp_rbuf_rd_idx')%sm.DISP_RBUF_SIZE]
            print('time=',x)
            sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_rd_idx']] = (sm.status_shm.buf[sm.STATUS_SHM_IDX['disp_rbuf_rd_idx']] + 1) % sm.DISP_RBUF_SIZE
            # x = x0/800
            doc.add_next_tick_callback(partial(update, x=[x], y=[y]))
            x0 += 1
        else:
            time.sleep(0.001)

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

probe_plot_panel, probe_graph = probe_plot.gen_probe_plot()

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

# probe_graph.node_renderer.data_source.selected.on_change('indices', select_callback)
probe_graph.edge_renderer.data_source.selected.on_change('indices', edge_select_callback)

# put the buttons and plot in a layout and add to the document

button_panel = column(Spacer(height=100), button_run, button_stop,button_run_power_calib, button_stop_power_calib, Spacer(width=500))
grid_panel = layout(
        children=[
            [button_panel, probe_plot_panel],
            [p],
        ],
        align='center'
        #sizing_mode='fixed',
    )

doc.add_root(grid_panel)

# doc.add_root(column(row(column(button_run, button_stop, align='center'), probe_plot_panel, align='center'), p, align='center'))
thread = Thread(target=blocking_task)
thread.start()

