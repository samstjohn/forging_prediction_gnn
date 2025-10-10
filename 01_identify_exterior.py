import logging
import pandas as pd



#######################################################################################################
# MAIN PROGRAM BEGINS HERE
#######################################################################################################

logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s:%(levelname)s: %(message)s')


root_dir = "/home/FORGING/input/"
csv_dir = "/home/FORGING/input/graph_csv/"
episode_file = root_dir + "graph_csv/episode_summary.csv"

episode_desc = pd.read_csv(episode_file)
logging.debug('Episode Description \n--------------------------------------\n %s\n', episode_desc)

simulation_counter = -1
simulation_columns = ['sim_counter', 'step_counter', 'graph_file']

simulation_files = pd.DataFrame(columns=simulation_columns)

nodes = []
node_attributes = []
edges = []


for i in range(len(episode_desc)):

    sim_time = episode_desc.iloc[i, 0]
    step_num = episode_desc.iloc[i, 1]

    # Determine if simulation is an initial step
    if sim_time < 0:
        simulation_counter += 1
        step_counter = 0
        sim_edges = []
        sim_nodes = []
        sim_node_attributes = []
        determine_exterior = 1
        logging.debug('Determining exterior for simulation %s', simulation_counter)
    else:
        simulation_counter += 1
        determine_exterior = 0
        logging.debug('Using prior exterior for simulation %s', simulation_counter)

    # Set graph_file variable and read in details
    if step_num < 10:
        graph_count = "00" + str(step_num)
    elif step_num < 100:
        graph_count = "0" + str(step_num)

    graph_file = "graph_" + graph_count + ".csv"
    exterior_graph_file = "ext_graph_" + graph_count + ".csv"
    graph_csv_source = csv_dir + graph_file
    exterior_graph_csv_destination = csv_dir + exterior_graph_file
    graph_df = pd.read_csv(graph_csv_source)
    workpiece_df = graph_df[(graph_df['ID'] == 2)]

    if determine_exterior == 1:
        logging.debug('Determining exterior for simulation %s', simulation_counter)
        exterior_nodes = []
        exterior_rows = []
        exterior_index = 0
        interior_counter = 0
        node_type_1_counter = 0
        node_type_2_counter = 0

        max_workpiece_x = workpiece_df['x'].max()
        min_workpiece_x = workpiece_df['x'].min()
        max_workpiece_y = workpiece_df['y'].max()
        min_workpiece_y = workpiece_df['y'].min()
        max_workpiece_z = workpiece_df['z'].max()
        min_workpiece_z = workpiece_df['z'].min()

        logging.debug('  Maximum X value for workpiece %s', max_workpiece_x)
        logging.debug('  Minimum X value for workpiece %s', min_workpiece_x)
        logging.debug('  Maximum Y value for workpiece %s', max_workpiece_y)
        logging.debug('  Minimum Y value for workpiece %s', min_workpiece_y)
        logging.debug('  Maximum Z value for workpiece %s', max_workpiece_z)
        logging.debug('  Minimum Z value for workpiece %s\n', min_workpiece_z)

        for index, graph_node in graph_df.iterrows():

            # Gather current node information
            current_node = graph_node['node']
            current_x = graph_node['x']
            current_y = graph_node['y']
            current_z = graph_node['z']
            current_neighbors_string = graph_node['neighbors']
            current_neighbors = graph_node['neighbors'].replace('[', '').replace(']', '').replace(',', '').split()
            current_num_of_neighbors = len(current_neighbors)
            current_node_type = graph_node['ID']

            # Determine interior / exterior based on number of neighbors for workpiece
            if current_node_type == 1:
                if current_num_of_neighbors < 6:
                    current_exterior = 1
                    node_type_1_counter += 1
                else:
                    current_exterior = 0

            # All node_type 2 nodes are exterior - this is the tool
            elif current_node_type == 2:
                current_exterior = 1
                node_type_2_counter += 1
                #if  (current_x == max_workpiece_x) or (current_x == min_workpiece_x) or \
                #    (current_y == max_workpiece_y) or (current_y == min_workpiece_y) or \
                #    (current_z == max_workpiece_z) or (current_z == min_workpiece_z):
                #    current_exterior = 1
                #    node_type_2_counter += 1
                #else:
                #    current_exterior = 0
            else:
                current_exterior = 0

            if current_exterior == 1:
                exterior_nodes.append(current_node)
                exterior_row = [current_node, current_neighbors_string, current_x, current_y, current_z, current_node_type]
                exterior_rows.append(exterior_row)
                exterior_index += 1
            else:
                interior_counter += 1

        logging.debug('  Count of exterior nodes %s', exterior_index)
        logging.debug('  Count of node type 1 exterior nodes %s', node_type_1_counter)
        logging.debug('  Count of node type 2 exterior nodes %s', node_type_2_counter)
        logging.debug('  Count of interior nodes %s\n', interior_counter)

        current_exterior_df = pd.DataFrame(exterior_rows)
        current_exterior_df.columns = ['node', 'neighbors', 'x', 'y', 'z', 'ID']

        logging.debug('  Saving output as CSV: %s', exterior_graph_csv_destination)
        current_exterior_df.to_csv(exterior_graph_csv_destination)


    else:
        logging.debug('  Prior count of exterior nodes: %s', len(exterior_rows))
        exterior_rows = []
        exterior_index = 0

        for index, graph_node in graph_df.iterrows():

            current_node = graph_node['node']
            current_x = graph_node['x']
            current_y = graph_node['y']
            current_z = graph_node['z']
            current_neighbors_string = graph_node['neighbors']
            current_node_type = graph_node['ID']

            if current_node in exterior_nodes:
                exterior_row = [current_node, current_neighbors_string, current_x, current_y, current_z, current_node_type]
                exterior_rows.append(exterior_row)
                exterior_index += 1

        logging.debug('  Count of exterior nodes %s', exterior_index)

        current_exterior_df = pd.DataFrame(exterior_rows)
        current_exterior_df.columns = ['node', 'neighbors', 'x', 'y', 'z', 'ID']

        logging.debug('  Saving output as CSV: %s', exterior_graph_csv_destination)
        current_exterior_df.to_csv(exterior_graph_csv_destination)



