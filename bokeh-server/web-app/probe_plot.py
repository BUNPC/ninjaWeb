import os
import configparser
from scipy import io
from bokeh.plotting import figure
from bokeh.models import GraphRenderer, Circle,  StaticLayoutProvider, MultiLine, NodesAndLinkedEdges, EdgesOnly

def gen_probe_plot():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    config_file_path = os.path.join(file_dir, '..','..', 'probes', 'Config.cfg')
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # load SD file
    probe_filename = config['Probe']['filename']
    sd_fname = os.path.join(file_dir, '..', '..', 'probes', probe_filename)

    # sd_fname = './web-app/fullhead_56x144_v2.SD.mat'

    sd_data = io.loadmat(sd_fname)

    src_pos = sd_data['SD']['SrcPos2D'][0][0]
    det_pos = sd_data['SD']['DetPos2D'][0][0]
    ml = sd_data['SD']['MeasList'][0][0]
    n_src = len(src_pos)
    n_det = len(det_pos)

    # create single wavelength measurement list
    if (len(ml)%2 != 0) or (max(ml[:,3]) != 2):
        print("Error: Measurement list not as expected.")
    ml_onewl = ml[:int(len(ml)/2), [0, 1]]

    # calculate square axis ranges
    min_c = min(src_pos.min(), det_pos.min())*1.08
    max_c = max(src_pos.max(), det_pos.max())*1.08

    # list the nodes
    node_indices = list(range(n_src+n_det))
    node_colors = ['red']*n_src + ['blue']*n_det
    sd_desc = [f"S{ii+1}" for ii in range(n_src)] + [f"D{ii+1}" for ii in range(n_det)]

    plot = figure(title="Probe Geometry", x_range=(min_c,max_c), y_range=(min_c,max_c), \
                  width=500, height=500, align='center', tools="tap,hover,examine", tooltips="@sd_desc")
    plot.grid.grid_line_color = None
    plot.axis.visible = False

    graph = GraphRenderer()

    # replace the node glyph with an ellipse
    # set its height, width, and fill_color
    graph.node_renderer.glyph = Circle(radius=0.02, fill_color="fill_color")

    # assign a palette to ``fill_color`` and add it to the data source
    graph.node_renderer.data_source.data = dict(
        index=node_indices,
        fill_color=node_colors,
        sd_desc=sd_desc)
    

    edge_glyph = MultiLine(line_color="slategrey", line_width=1, line_alpha=0.9)
    graph.edge_renderer.glyph = edge_glyph
    graph.edge_renderer.selection_glyph = edge_glyph.clone(line_color="grey", line_width=3, line_alpha=1)
    # add the rest of the assigned values to the data source
    graph.edge_renderer.data_source.data = dict(
        start=list(ml_onewl[:, 0] -1),
        end=list(ml_onewl[:, 1] + n_src -1))
    
    graph.selection_policy = EdgesOnly() #NodesAndLinkedEdges()#NodesOnly()
    #graph.inspection_policy = NodesAndLinkedEdges()

    # convert the ``x`` and ``y`` lists into a dictionary of 2D-coordinates
    # and assign each entry to a node on the ``node_indices`` list
    graph_layout = dict(zip(node_indices, zip(list(src_pos[:,0])+list(det_pos[:,0]), list(src_pos[:,1])+list(det_pos[:,1]))))

    # use the provider model to supply coourdinates to the graph
    graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)

    # render the graph
    plot.renderers.append(graph)

    return plot, graph
