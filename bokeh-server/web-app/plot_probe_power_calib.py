import os
import configparser
from scipy import io
import numpy as np
import bokeh
from bokeh.plotting import figure
from bokeh.models import (GraphRenderer, Circle, StaticLayoutProvider,
                          MultiLine, NodesAndLinkedEdges, EdgesOnly,
                          ColumnDataSource, LabelSet)


def gen_probe_plot(sd_data):
    # file_dir = os.path.dirname(os.path.realpath(__file__))
    # config_file_path = os.path.join(file_dir, '..', '..', 'probes', 'Config.cfg')
    # config = configparser.ConfigParser()
    # config.read(config_file_path)
    #
    # # load SD file
    # probe_filename = config['Probe']['filename']
    # sd_fname = os.path.join(file_dir, '..', '..', 'probes', probe_filename)

    # sd_fname = './web-app/fullhead_56x144_v2.SD.mat'

    # sd_data = io.loadmat(sd_fname)

    src_pos = sd_data['SD']['SrcPos2D'][0][0]
    det_pos = sd_data['SD']['DetPos2D'][0][0]
    ml = sd_data['SD']['MeasList'][0][0]
    ml_length = len(np.where(ml[:, 3] == 1)[0])
    n_src = len(src_pos)
    n_det = len(det_pos)

    # create single wavelength measurement list
    if (len(ml) % 2 != 0) or (max(ml[:, 3]) != 2):
        print("Error: Measurement list not as expected.")
    ml_onewl = ml[:int(len(ml) / 2), [0, 1]]

    # calculate square axis ranges
    min_c = min(src_pos.min(), det_pos.min()) * 1.08
    max_c = max(src_pos.max(), det_pos.max()) * 1.08

    # list the nodes
    node_indices = list(range(n_src + n_det))
    node_colors = ['red'] * n_src + ['blue'] * n_det
    sd_desc = [f"S{ii + 1}" for ii in range(n_src)] + [f"D{ii + 1}" for ii in range(n_det)]
    # node_alpha = [0] * len(node_indices)  # Initialize all nodes as invisible

    plot = figure(title="Signal Level", x_range=(min_c, max_c), y_range=(min_c, max_c),
                  width=500, height=500, align='center', tools="tap,hover,examine", tooltips="@sd_desc")
    plot.grid.grid_line_color = None
    plot.axis.visible = False

    graph = GraphRenderer()

    # replace the node glyph with an ellipse
    # set its height, width, and fill_color
    graph.node_renderer.glyph = Circle(radius=0.01, fill_color="fill_color", fill_alpha='alpha', line_alpha=0)

    # assign a palette to ``fill_color`` and add it to the data source
    # graph.node_renderer.data_source.data = dict(
    #     index=node_indices,
    #     fill_color=node_colors,
    #     sd_desc=sd_desc,
    #     alpha=[0] * len(node_indices))
    node_source = ColumnDataSource(data=dict(
        index=node_indices,
        fill_color=node_colors,
        sd_desc=sd_desc,
        alpha=[0.0] * len(node_indices)  # <-- Nodes initially invisible
    ))
    graph.node_renderer.data_source = node_source

    edge_glyph = MultiLine(line_color="line_color", line_width=1, line_alpha='alpha')
    graph.edge_renderer.glyph = edge_glyph
    graph.edge_renderer.selection_glyph = edge_glyph.clone(line_color="grey", line_width=3, line_alpha='alpha')
    # edge_alpha = [0] * len(ml_onewl)  # Initialize all edges as invisible
    # add the rest of the assigned values to the data source
    # graph.edge_renderer.data_source.data = dict(
    #     start=list(ml_onewl[:, 0] - 1),
    #     end=list(ml_onewl[:, 1] + n_src - 1),
    #     alpha=[0] * len(ml_onewl),
    #     line_color=["red"] * len(ml_onewl))
    edge_source = ColumnDataSource(data=dict(
        # Ensure start/end indices are correct (0-based)
        start=list(ml_onewl[:, 0] - 1) if ml_length > 0 else [],
        end=list(ml_onewl[:, 1] + n_src - 1) if ml_length > 0 else [],
        alpha=[0.0] * ml_length,  # <-- Edges initially invisible
        line_color=["grey"] * ml_length  # Default color when invisible/dim
    ))
    graph.edge_renderer.data_source = edge_source

    graph.selection_policy = EdgesOnly()  # NodesAndLinkedEdges()#NodesOnly()
    # graph.inspection_policy = NodesAndLinkedEdges()

    # convert the ``x`` and ``y`` lists into a dictionary of 2D-coordinates
    # and assign each entry to a node on the ``node_indices`` list
    node_x = list(src_pos[:, 0]) + list(det_pos[:, 0])
    node_y = list(src_pos[:, 1]) + list(det_pos[:, 1])
    graph_layout = dict(
        zip(node_indices, zip(node_x, node_y)))

    # use the provider model to supply coordinates to the graph
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    # Add labels (Numbers)
    # label_alpha = [0] * len(sd_desc)  # Initialize all labels as invisible
    label_source = ColumnDataSource(data=dict(x=node_x,
                                              y=node_y,
                                              labels=sd_desc,
                                              alpha=[0] * len(sd_desc),
                                              label_color=node_colors))
    labels = LabelSet(x='x', y='y', text='labels', source=label_source,
                      text_align='center', text_baseline='middle', text_color='label_color', text_font_size="12pt",
                      text_alpha='alpha',
                      text_font_style='bold')

    # render the graph
    plot.renderers.append(graph)
    plot.add_layout(labels)
    print('inside plot probe -11')

    return plot, graph, labels, label_source, node_source, edge_source
