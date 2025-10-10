import logging
import os
import numpy as np
import pandas as pd
import itertools
import math
import open3d as o3d

#######################################################################################################
# GENERATE FACES
def generate_faces(nodes, edges, node_attributes):

    for i in edges:
        #logging.debug('i in edges: %s', i)

        cell_elements = []

        # node_counter is the node value, starting with 1
        node_counter = 0

        # edge_count values
        one_edge_count = 0
        two_edge_count = 0
        three_edge_count = 0
        four_edge_count = 0
        five_edge_count = 0
        six_edge_count = 0

        # Build list of tool nodes and workpiece nodes
        tool_nodes = []
        workpiece_nodes = []
        while node_counter < len(nodes[0][1]):
            center_node = nodes[0][1][node_counter]
            center_node_id = node_attributes[0][1][node_counter][3]
            if (center_node_id == 2):
                tool_nodes.append(center_node)
            else:
                workpiece_nodes.append(center_node)

            node_counter += 1



        node_counter = 0
        while node_counter < len(nodes[0][1]):
            edge_counter = 0
            branch_nodes = []

            center_node = nodes[0][1][node_counter]

            for j in i[1]:
                if center_node in j:
                    for p in j:
                        if (p != center_node) and (p in nodes[0][1]):
                            branch_nodes.append(p)
                    edge_counter += 1


            # Now we have the center node and all its branch nodes
            # we build triangular mesh connectivity based on distance between branch nodes
            bn_combinations = list(itertools.combinations(branch_nodes, 2))
            euclidian_distances = []

            for bn_combination in bn_combinations:

                if (bn_combination[0] in nodes[0][1]) and (bn_combination[1] in nodes[0][1]):
                    bn_index_1 = nodes[0][1].index(bn_combination[0])
                    bn_index_2 = nodes[0][1].index(bn_combination[1])

                    try:
                        P = node_attributes[0][1][bn_index_1][:3]
                    except:
                        print("Error on P")
                        print("node_attributes[0]: ", node_attributes[0])
                        exit(1)

                    try:
                        Q = node_attributes[0][1][bn_index_2][:3]
                    except:
                        print("Error on Q")
                        print("i[0]: ", i[0])
                        print("node_attributes[0][1]: ", node_attributes[0][1][bn_index_2])
                        exit(1)

                    euclidian_dist = math.dist(P, Q)
                    euclidian_distances.append(euclidian_dist)

            faces = []
            face_elements = []

            #print("Face 1: ", center_node, bn_combinations[euclidian_distances.index(min(euclidian_distances))])
            cell_element = []
            cell_element.append(center_node)
            cell_element.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))][0])
            cell_element.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))][1])
            cell_element.sort()
            if cell_element not in cell_elements:
                cell_elements.append(cell_element)

            faces.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))])
            for item in bn_combinations[euclidian_distances.index(min(euclidian_distances))]:
                face_elements.append(item)
            del bn_combinations[euclidian_distances.index(min(euclidian_distances))]
            del euclidian_distances[euclidian_distances.index(min(euclidian_distances))]

            #print("Face 2: ", center_node, bn_combinations[euclidian_distances.index(min(euclidian_distances))])
            cell_element = []
            cell_element.append(center_node)
            cell_element.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))][0])
            cell_element.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))][1])
            cell_element.sort()
            if cell_element not in cell_elements:
                cell_elements.append(cell_element)

            faces.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))])
            for item in bn_combinations[euclidian_distances.index(min(euclidian_distances))]:
                face_elements.append(item)
            del bn_combinations[euclidian_distances.index(min(euclidian_distances))]
            del euclidian_distances[euclidian_distances.index(min(euclidian_distances))]

            for element in set(face_elements):
                if face_elements.count(element) >= 2:
                    updated_bn_combinations = []
                    updated_euclidian_distances = []
                    for bn_combination in bn_combinations:
                        if bn_combination.count(element) == 0:
                            include_index = bn_combinations.index(bn_combination)
                            updated_bn_combinations.append(bn_combination)
                            updated_euclidian_distances.append(euclidian_distances[include_index])

                    bn_combinations = updated_bn_combinations
                    euclidian_distances = updated_euclidian_distances

            #print("Face 3: ", center_node, bn_combinations[euclidian_distances.index(min(euclidian_distances))])
            cell_element = []
            cell_element.append(center_node)
            cell_element.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))][0])
            cell_element.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))][1])
            cell_element.sort()
            if cell_element not in cell_elements:
                cell_elements.append(cell_element)

            faces.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))])
            for item in bn_combinations[euclidian_distances.index(min(euclidian_distances))]:
                face_elements.append(item)
            del bn_combinations[euclidian_distances.index(min(euclidian_distances))]
            del euclidian_distances[euclidian_distances.index(min(euclidian_distances))]

            for element in set(face_elements):
                if face_elements.count(element) >= 2:
                    updated_bn_combinations = []
                    updated_euclidian_distances = []
                    for bn_combination in bn_combinations:
                        if bn_combination.count(element) == 0:
                            include_index = bn_combinations.index(bn_combination)
                            updated_bn_combinations.append(bn_combination)
                            updated_euclidian_distances.append(euclidian_distances[include_index])

                    bn_combinations = updated_bn_combinations
                    euclidian_distances = updated_euclidian_distances

            if len(bn_combinations) > 0:
                #print("Face 4: ", center_node, bn_combinations[euclidian_distances.index(min(euclidian_distances))])
                cell_element = []
                cell_element.append(center_node)
                cell_element.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))][0])
                cell_element.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))][1])
                cell_element.sort()
                if cell_element not in cell_elements:
                    cell_elements.append(cell_element)

                faces.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))])
                for item in bn_combinations[euclidian_distances.index(min(euclidian_distances))]:
                    face_elements.append(item)
                del bn_combinations[euclidian_distances.index(min(euclidian_distances))]
                del euclidian_distances[euclidian_distances.index(min(euclidian_distances))]

                for element in set(face_elements):
                    if face_elements.count(element) >= 2:
                        updated_bn_combinations = []
                        updated_euclidian_distances = []
                        for bn_combination in bn_combinations:
                            if bn_combination.count(element) == 0:
                                include_index = bn_combinations.index(bn_combination)
                                updated_bn_combinations.append(bn_combination)
                                updated_euclidian_distances.append(euclidian_distances[include_index])

                        bn_combinations = updated_bn_combinations
                        euclidian_distances = updated_euclidian_distances

            if len(bn_combinations) > 0:
                #print("Face 5: ", center_node, bn_combinations[euclidian_distances.index(min(euclidian_distances))])
                cell_element = []
                cell_element.append(center_node)
                cell_element.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))][0])
                cell_element.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))][1])
                cell_element.sort()
                if cell_element not in cell_elements:
                    cell_elements.append(cell_element)

                faces.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))])
                for item in bn_combinations[euclidian_distances.index(min(euclidian_distances))]:
                    face_elements.append(item)
                del bn_combinations[euclidian_distances.index(min(euclidian_distances))]
                del euclidian_distances[euclidian_distances.index(min(euclidian_distances))]

                for element in set(face_elements):
                    if face_elements.count(element) >= 2:
                        updated_bn_combinations = []
                        updated_euclidian_distances = []
                        for bn_combination in bn_combinations:
                            if bn_combination.count(element) == 0:
                                include_index = bn_combinations.index(bn_combination)
                                updated_bn_combinations.append(bn_combination)
                                updated_euclidian_distances.append(euclidian_distances[include_index])

                        bn_combinations = updated_bn_combinations
                        euclidian_distances = updated_euclidian_distances

            if len(bn_combinations) > 0:
                #print("Face 6: ", center_node, bn_combinations[euclidian_distances.index(min(euclidian_distances))])
                cell_element = []
                cell_element.append(center_node)
                cell_element.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))][0])
                cell_element.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))][1])
                cell_element.sort()
                if cell_element not in cell_elements:
                    cell_elements.append(cell_element)

                faces.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))])
                for item in bn_combinations[euclidian_distances.index(min(euclidian_distances))]:
                    face_elements.append(item)
                del bn_combinations[euclidian_distances.index(min(euclidian_distances))]
                del euclidian_distances[euclidian_distances.index(min(euclidian_distances))]

                for element in set(face_elements):
                    if face_elements.count(element) >= 2:
                        updated_bn_combinations = []
                        updated_euclidian_distances = []
                        for bn_combination in bn_combinations:
                            if bn_combination.count(element) == 0:
                                include_index = bn_combinations.index(bn_combination)
                                updated_bn_combinations.append(bn_combination)
                                updated_euclidian_distances.append(euclidian_distances[include_index])

                        bn_combinations = updated_bn_combinations
                        euclidian_distances = updated_euclidian_distances

            if len(bn_combinations) > 0:
                #print("Face 7: ", center_node, bn_combinations[euclidian_distances.index(min(euclidian_distances))])
                cell_element = []
                cell_element.append(center_node)
                cell_element.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))][0])
                cell_element.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))][1])
                cell_element.sort()
                if cell_element not in cell_elements:
                    cell_elements.append(cell_element)

                faces.append(bn_combinations[euclidian_distances.index(min(euclidian_distances))])
                for item in bn_combinations[euclidian_distances.index(min(euclidian_distances))]:
                    face_elements.append(item)

            if edge_counter == 1:
                one_edge_count += 1
            elif edge_counter == 2:
                two_edge_count += 1
            elif edge_counter == 3:
                three_edge_count += 1
            elif edge_counter == 4:
                four_edge_count += 1
            elif edge_counter == 5:
                five_edge_count += 1
            elif edge_counter == 6:
                six_edge_count += 1


            if center_node in workpiece_nodes:

                #print("Creating workpiece faces.")
                connected_combinations = list(itertools.combinations(tool_nodes, 2))
                euclidian_distances = []

                # UPDATE TO REDUCE COMBINATIONS TO THE x CLOSEST NODES
                cn_pairs = []
                cn_dist = []
                for tool_node in tool_nodes:
                    cn_pairs.append([center_node, tool_node])

                # FIND EUCLIDIAN DISTANCE BETWEEN NODES; ONLY KEEP TOP XX
                for cn_pair in cn_pairs:

                    if (cn_pair[0] in nodes[0][1]) and (cn_pair[1] in nodes[0][1]):
                        cn_index_1 = nodes[0][1].index(cn_pair[0])
                        cn_index_2 = nodes[0][1].index(cn_pair[1])

                        try:
                            P = node_attributes[0][1][cn_index_1][:3]
                        except:
                            print("Error on P")
                            print("node_attributes[0]: ", node_attributes[0])
                            exit(1)

                        try:
                            Q = node_attributes[0][1][cn_index_2][:3]
                        except:
                            print("Error on Q")
                            print("i[0]: ", i[0])
                            print("node_attributes[0][1]: ", node_attributes[0][1][cn_index_2])
                            exit(1)

                        euclidian_dist = math.dist(P, Q)
                        euclidian_distances.append(euclidian_dist)

                connected_graph_nodes = []
                connected_graph_node_counter = 0



                while connected_graph_node_counter < 8:
                    #print("connected graph node counter: ", connected_graph_node_counter)
                    try:
                        cg_node = cn_pairs[euclidian_distances.index(min(euclidian_distances))][1]
                        connected_graph_nodes.append(cg_node)
                        del cn_pairs[euclidian_distances.index(min(euclidian_distances))]
                        del euclidian_distances[euclidian_distances.index(min(euclidian_distances))]

                    except:
                        print("Error on cg_node")
                        print("cn_pairs: ", cn_pairs)
                        print("euclidian_distances: ", euclidian_distances)

                    connected_graph_node_counter += 1

                #logging.debug('Connected graph nodes %s', connected_graph_nodes)
                cn_combinations = list(itertools.combinations(connected_graph_nodes, 2))
                #logging.debug('Connected graph node combinations %s', cn_combinations)

                # DETERMINE TRIANGLES TO USE IN MESH
                euclidian_distances = []

                for connected_combination in cn_combinations:

                    if (connected_combination[0] in nodes[0][1]) and (connected_combination[1] in nodes[0][1]):
                        cn_index_1 = nodes[0][1].index(connected_combination[0])
                        cn_index_2 = nodes[0][1].index(connected_combination[1])
                        #print("CN Index 1: ", cn_index_1)
                        #print("CN Index 2: ", cn_index_2)

                        try:
                            P = node_attributes[0][1][cn_index_1][:3]
                        except:
                            print("Error on P")
                            print("node_attributes[0]: ", node_attributes[0])
                            exit(1)

                        try:
                            Q = node_attributes[0][1][cn_index_2][:3]
                        except:
                            print("Error on Q")
                            print("i[0]: ", i[0])
                            print("node_attributes[0][1]: ", node_attributes[0][1][cn_index_2])
                            exit(1)

                        euclidian_dist = math.dist(P, Q)
                        euclidian_distances.append(euclidian_dist)

                #print("Connected Face 1: ", center_node, cn_combinations[euclidian_distances.index(min(euclidian_distances))])
                #print("Euclidian Distances: ", min(euclidian_distances))
                cell_element = []
                cell_element.append(center_node)
                cell_element.append(cn_combinations[euclidian_distances.index(min(euclidian_distances))][0])
                cell_element.append(cn_combinations[euclidian_distances.index(min(euclidian_distances))][1])
                cell_element.sort()
                if cell_element not in cell_elements:
                    cell_elements.append(cell_element)

                faces.append(cn_combinations[euclidian_distances.index(min(euclidian_distances))])
                for item in cn_combinations[euclidian_distances.index(min(euclidian_distances))]:
                    face_elements.append(item)
                del cn_combinations[euclidian_distances.index(min(euclidian_distances))]
                del euclidian_distances[euclidian_distances.index(min(euclidian_distances))]

                #print("Connected Face 2: ", center_node, cn_combinations[euclidian_distances.index(min(euclidian_distances))])
                #print("Euclidian Distances: ", min(euclidian_distances))
                cell_element = []
                cell_element.append(center_node)
                cell_element.append(cn_combinations[euclidian_distances.index(min(euclidian_distances))][0])
                cell_element.append(cn_combinations[euclidian_distances.index(min(euclidian_distances))][1])
                cell_element.sort()
                if cell_element not in cell_elements:
                    cell_elements.append(cell_element)

                faces.append(cn_combinations[euclidian_distances.index(min(euclidian_distances))])
                for item in cn_combinations[euclidian_distances.index(min(euclidian_distances))]:
                    face_elements.append(item)
                del cn_combinations[euclidian_distances.index(min(euclidian_distances))]
                del euclidian_distances[euclidian_distances.index(min(euclidian_distances))]

                #print("Connected Face 3: ", center_node, cn_combinations[euclidian_distances.index(min(euclidian_distances))])
                #print("Euclidian Distances: ", min(euclidian_distances))
                cell_element = []
                cell_element.append(center_node)
                cell_element.append(cn_combinations[euclidian_distances.index(min(euclidian_distances))][0])
                cell_element.append(cn_combinations[euclidian_distances.index(min(euclidian_distances))][1])
                cell_element.sort()
                if cell_element not in cell_elements:
                    cell_elements.append(cell_element)

                faces.append(cn_combinations[euclidian_distances.index(min(euclidian_distances))])
                for item in cn_combinations[euclidian_distances.index(min(euclidian_distances))]:
                    face_elements.append(item)
                del cn_combinations[euclidian_distances.index(min(euclidian_distances))]
                del euclidian_distances[euclidian_distances.index(min(euclidian_distances))]

                #print("Connected Face 4: ", center_node, cn_combinations[euclidian_distances.index(min(euclidian_distances))])
                #print("Euclidian Distances: ", min(euclidian_distances))
                cell_element = []
                cell_element.append(center_node)
                cell_element.append(cn_combinations[euclidian_distances.index(min(euclidian_distances))][0])
                cell_element.append(cn_combinations[euclidian_distances.index(min(euclidian_distances))][1])
                cell_element.sort()
                if cell_element not in cell_elements:
                    cell_elements.append(cell_element)

                faces.append(cn_combinations[euclidian_distances.index(min(euclidian_distances))])
                for item in cn_combinations[euclidian_distances.index(min(euclidian_distances))]:
                    face_elements.append(item)
                del cn_combinations[euclidian_distances.index(min(euclidian_distances))]
                del euclidian_distances[euclidian_distances.index(min(euclidian_distances))]


            node_counter += 1

        #print(len(cell_elements))
        #exit(0)

        return cell_elements

#######################################################################################################
# MAIN PROGRAM BEGINS HERE
#######################################################################################################

logging.basicConfig(level = logging.INFO, format = '%(asctime)s:%(levelname)s: %(message)s')

root_dir = "/home/FORGING/input/"
csv_dir = "/home/FORGING/input/graph_csv/"
episode_file = root_dir + "graph_csv/episode_summary.csv"

episode_desc = pd.read_csv(episode_file)
logging.info('Episode Description \n--------------------------------------\n %s\n', episode_desc)

simulation_counter = -1
simulation_columns = ['sim_counter', 'step_counter', 'graph_file']

display_graphs = 0

# SET THE NUMBER OF EDGES FROM EACH TOOL NODE TO THE WORKPIECE
tool_to_wp_connections = 2
max_tool_to_wp_connections = 10

create_tool_to_wp = 1
create_wp_to_tool = 1


# BUILD FOR A SERIES OF TOOL TO WP CONNECTIONS
while tool_to_wp_connections <= max_tool_to_wp_connections:
    output_dir = "/home/FORGING/input/graph_csv/world_" + str(tool_to_wp_connections) + "/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    simulation_files = pd.DataFrame(columns=simulation_columns)

    for i in range(len(episode_desc)):

        nodes = []
        node_attributes = []
        edges = []

        sim_time = episode_desc.iloc[i, 0]
        step_num = episode_desc.iloc[i, 1]

        logging.info('  Current sim_time: %s', sim_time)

        if step_num < 10:
            graph_count = "00" + str(step_num)
        elif step_num < 100:
            graph_count = "0" + str(step_num)

        graph_file = "graph_" + graph_count + ".csv"
        full_faces_file = "full_faces_" + graph_count + ".csv"
        full_faces_csv_destination = output_dir + full_faces_file

        all_vertices_file = "all_vertices_" + graph_count + ".csv"
        all_vertices_csv_destination = output_dir + all_vertices_file

        workpiece_surface_faces_file = "workpiece_surface_faces_" + graph_count + ".csv"
        workpiece_surface_faces_csv_destination = output_dir + workpiece_surface_faces_file

        tool_surface_faces_file = "tool_surface_faces_" + graph_count + ".csv"
        tool_surface_faces_csv_destination = output_dir + tool_surface_faces_file

        tool_to_wp_connection_faces_file = "tool_to_wp_world_faces_" + graph_count + ".csv"
        tool_to_wp_connection_faces_csv_destination = output_dir + tool_to_wp_connection_faces_file

        wp_to_tool_connection_faces_file = "wp_to_tool_world_faces_" + graph_count + ".csv"
        wp_to_tool_connection_faces_csv_destination = output_dir + wp_to_tool_connection_faces_file

        if sim_time < 0:
            simulation_counter += 1
            step_counter = 0
            workpiece_upper_face_sim_edges = []
            workpiece_lower_face_sim_edges = []
            workpiece_front_face_sim_edges = []
            workpiece_back_face_sim_edges = []
            workpiece_right_face_sim_edges = []
            workpiece_left_face_sim_edges = []
            sim_edges = []
            sim_nodes = []
            sim_node_attributes = []
            create_edge_list = 1
        else:
            create_edge_list = 0
            #logging.debug('Saving cell faces to: %s', full_faces_csv_destination)
            #cell_faces_df.to_csv(full_faces_csv_destination)



        simulation_files = pd.concat([simulation_files, pd.DataFrame([[simulation_counter, step_counter, graph_file]], columns=simulation_columns)], ignore_index=True)

        graph_csv_source = csv_dir + graph_file
        graph_df = pd.read_csv(graph_csv_source)

        logging.info('    graph_df shape: %s', graph_df.shape)
        logging.debug('    Data review: graph_df %s', graph_df)

        if create_edge_list == 0:
            logging.info('    workpiece_face_nodes length: %s', len(workpiece_face_nodes))
            logging.debug('    Data review: workpiece_face_nodes %s', workpiece_face_nodes)

            workpiece_vertices = []
            for node in workpiece_face_nodes:
                vertex = []
                node_x = float(graph_df.loc[graph_df['node'] == node]['x'])
                node_y = float(graph_df.loc[graph_df['node'] == node]['y'])
                node_z = float(graph_df.loc[graph_df['node'] == node]['z'])

                vertex.append(node_x)
                vertex.append(node_y)
                vertex.append(node_z)

                workpiece_vertices.append(vertex)


            logging.info('    tool_nodes length: %s', len(tool_nodes))
            logging.debug('    Data review: tool_nodes %s', tool_nodes)

            tool_vertices = []
            for node in tool_nodes:
                vertex = []
                node_x = float(graph_df.loc[graph_df['node'] == node]['x'])
                node_y = float(graph_df.loc[graph_df['node'] == node]['y'])
                node_z = float(graph_df.loc[graph_df['node'] == node]['z'])

                vertex.append(node_x)
                vertex.append(node_y)
                vertex.append(node_z)

                tool_vertices.append(vertex)

            workpiece_vertices_df = pd.DataFrame(workpiece_vertices)
            workpiece_vertices_df.columns = ['x', 'y', 'z']
            workpiece_vertices_df['ID'] = 1

            tool_vertices_df = pd.DataFrame(tool_vertices)
            tool_vertices_df.columns = ['x', 'y', 'z']
            tool_vertices_df['ID'] = 2

            all_vertices_df = pd.concat([workpiece_vertices_df, tool_vertices_df])
            all_vertices_df = all_vertices_df.reset_index(drop=True)

            logging.info('    all_vertices_df shape: %s', all_vertices_df.shape)
            logging.debug('    Data review: all_vertices_df %s', all_vertices_df)

            logging.debug('Saving all vertices faces to: %s', all_vertices_csv_destination)
            all_vertices_df.to_csv(all_vertices_csv_destination)

            logging.debug('Saving workpiece surface faces to: %s', workpiece_surface_faces_csv_destination)
            workpiece_surface_faces_df.to_csv(workpiece_surface_faces_csv_destination)

            logging.debug('Saving tool surface faces to: %s', tool_surface_faces_csv_destination)
            tool_surface_faces_df.to_csv(tool_surface_faces_csv_destination)

            if build_connections == 1:
                if create_tool_to_wp == 1:
                    logging.debug('Saving tool to workpiece connection faces to: %s', tool_to_wp_connection_faces_csv_destination)
                    tool_to_wp_faces_df.to_csv(tool_to_wp_connection_faces_csv_destination)
                if create_wp_to_tool == 1:
                    logging.debug('Saving workpiece to tool connection faces to: %s', wp_to_tool_connection_faces_csv_destination)
                    wp_to_tool_faces_df.to_csv(wp_to_tool_connection_faces_csv_destination)


        else:
            workpiece_node_details_df = graph_df.loc[graph_df['ID'] == 1]
            workpiece_node_xyz = workpiece_node_details_df[['x','y','z']].to_numpy()
            workpiece_node_z = workpiece_node_details_df[['z']].to_numpy()


            ####################################################################################################################
            # BUILD WORKPIECE POINT CLOUD AND MESH
            ####################################################################################################################
            max_workpiece_z = workpiece_node_details_df['z'].max()
            min_workpiece_z = workpiece_node_details_df['z'].min()
            max_workpiece_x = workpiece_node_details_df['x'].max()
            min_workpiece_x = workpiece_node_details_df['x'].min()
            max_workpiece_y = workpiece_node_details_df['y'].max()
            min_workpiece_y = workpiece_node_details_df['y'].min()

            workpiece_upper_face_nodes = workpiece_node_details_df.loc[workpiece_node_details_df['z'] > (max_workpiece_z - 2)]
            workpiece_lower_face_nodes = workpiece_node_details_df.loc[workpiece_node_details_df['z'] < (min_workpiece_z + 2)]
            workpiece_front_face_nodes = workpiece_node_details_df.loc[workpiece_node_details_df['x'] > (max_workpiece_x - 2)]
            workpiece_back_face_nodes = workpiece_node_details_df.loc[workpiece_node_details_df['x'] < (min_workpiece_x + 2)]
            workpiece_right_face_nodes = workpiece_node_details_df.loc[workpiece_node_details_df['y'] > (max_workpiece_y - 2)]
            workpiece_left_face_nodes = workpiece_node_details_df.loc[workpiece_node_details_df['y'] < (min_workpiece_y + 2)]

            ####################################################################################################################
            # BUILD UPPER FACE POINT CLOUD
            workpiece_face_upper_nodes = []
            workpiece_face_upper_nodes.append(workpiece_upper_face_nodes['node'].tolist())

            logging.info('    workpiece_face_upper_nodes[0] length: %s', len(workpiece_face_upper_nodes[0]))
            logging.debug('    Data review: workpiece_face_upper_nodes[0] %s', workpiece_face_upper_nodes[0])


            # BUILD THE UPPER FACE CONNECTED POINTS FOR TRIANGLES
            wf_upper_tri = []
            for wf_node in workpiece_face_upper_nodes[0]:
                wf_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['x'])
                wf_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['y'])
                wf_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['z'])
                wf_upper = []
                wf_upper.append(wf_node)

                for node in workpiece_face_upper_nodes[0]:
                    if (wf_node != node) and node not in wf_upper:
                        node_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['x'])
                        node_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['y'])
                        node_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['z'])

                        x_diff = abs(wf_x - node_x)
                        y_diff = abs(wf_y - node_y)
                        z_diff = abs(wf_z - node_z)

                        # THIS IS FOR UPPER AND LOWER
                        # THESE ARE DEFINED BY Z LOCATION
                        # IF X AND Z ARE ON THE SAME LINE AND Y IS DIFFERENT
                        if (x_diff < 3)  \
                                and (((y_diff < 5) and (y_diff >= 2)) and ((wf_y > 0 and node_y > 0) or (wf_y < 0 and node_y < 0) or ((wf_y + node_y) < 1))) \
                                and ((z_diff < 2) and ((wf_z > 0 and node_z > 0) or (wf_z < 0 and node_z < 0))):
                            wf_upper.append(node)
                        # IF Y AND Z ARE ON THE SAME LINE AND X IS DIFFERENT
                        elif ((x_diff < 7) and (x_diff >= 3))  \
                                and ((y_diff < 2) and ((wf_y > 0 and node_y > 0) or (wf_y < 0 and node_y < 0))) \
                                and ((z_diff < 2) and ((wf_z > 0 and node_z > 0) or (wf_z < 0 and node_z < 0))):
                            wf_upper.append(node)

                # FROM CONNECTED POINTS CREATE COMBINATIONS FOR FACES
                wf_upper_combinations = list(itertools.combinations(wf_upper, 3))

                logging.debug('    wf_upper_combinations length: %s', len(wf_upper_combinations))
                logging.debug('    Data review: wf_upper_combinations %s', wf_upper_combinations)

                for combination in wf_upper_combinations:
                    node_1_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['x'])
                    node_2_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['x'])
                    node_3_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['x'])

                    node_1_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['y'])
                    node_2_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['y'])
                    node_3_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['y'])

                    node_1_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['z'])
                    node_2_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['z'])
                    node_3_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['z'])

                    if ((abs(node_1_x - node_2_x) < 1) and abs((node_2_x - node_3_x) < 1)) and not ((abs(node_1_y - node_2_y) < 1) and abs((node_2_y - node_3_y) < 1)):
                        wf_upper_tri.append(combination)
                    elif ((abs(node_1_y - node_2_y) < 1) and abs((node_2_y - node_3_y) < 1)) and not ((abs(node_1_x - node_2_x) < 1) and abs((node_2_x - node_3_x) < 1)):
                        wf_upper_tri.append(combination)

            logging.info('    wf_upper_tri length: %s', len(wf_upper_tri))
            logging.debug('    Data review: wf_upper_tri %s', wf_upper_tri)

            if display_graphs == 1:
                wf_upper_vertices = []
                for node in workpiece_face_upper_nodes[0]:
                    vertex = []
                    node_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['x'])
                    node_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['y'])
                    node_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['z'])

                    vertex.append(node_x)
                    vertex.append(node_y)
                    vertex.append(node_z)

                    wf_upper_vertices.append(vertex)

                workpiece_face_nodes = workpiece_face_upper_nodes[0]

                workpiece_upper_faces = []
                for tri in wf_upper_tri:
                    new_tri = []
                    for node in tri:
                        node_locator = workpiece_face_nodes.index(node)
                        new_tri.append(node_locator)
                    workpiece_upper_faces.append(new_tri)

                # DISPLAY POINT CLOUD
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(wf_upper_vertices)
                o3d.visualization.draw_geometries([pcd], window_name="Workpiece Upper Point Cloud")

                # DISPLAY MESH AND TRIANGULAR FACES
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(wf_upper_vertices)
                mesh.triangles = o3d.utility.Vector3iVector(workpiece_upper_faces)
                o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, window_name="Workpiece Upper Mesh")

                # DISPLAY LINESET - VISUALIZATION OF MESH WITHOUT FACES
                lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
                o3d.visualization.draw_geometries([lineset], window_name="Workpiece Upper Lineset")

            ####################################################################################################################
            # BUILD LOWER FACE POINT CLOUD
            workpiece_face_lower_nodes = []
            workpiece_face_lower_nodes.append(workpiece_lower_face_nodes['node'].tolist())

            logging.info('    workpiece_face_lower_nodes[0] length: %s', len(workpiece_face_lower_nodes[0]))
            logging.debug('    Data review: workpiece_face_lower_nodes[0] %s', workpiece_face_lower_nodes[0])

            # BUILD THE LOWER FACE CONNECTED POINTS FOR TRIANGLES
            wf_lower_tri = []
            for wf_node in workpiece_face_lower_nodes[0]:
                wf_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['x'])
                wf_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['y'])
                wf_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['z'])
                wf_lower = []
                wf_lower.append(wf_node)

                for node in workpiece_face_lower_nodes[0]:
                    if (wf_node != node) and node not in wf_lower:
                        node_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['x'])
                        node_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['y'])
                        node_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['z'])

                        x_diff = abs(wf_x - node_x)
                        y_diff = abs(wf_y - node_y)
                        z_diff = abs(wf_z - node_z)

                        # THIS IS FOR UPPER AND LOWER
                        # THESE ARE DEFINED BY Z LOCATION
                        # IF X AND Z ARE ON THE SAME LINE AND Y IS DIFFERENT
                        if (x_diff < 3)  \
                                and (((y_diff < 5) and (y_diff >= 2)) and ((wf_y > 0 and node_y > 0) or (wf_y < 0 and node_y < 0) or ((wf_y + node_y) < 1))) \
                                and ((z_diff < 2) and ((wf_z > 0 and node_z > 0) or (wf_z < 0 and node_z < 0))):
                            wf_lower.append(node)
                        # IF Y AND Z ARE ON THE SAME LINE AND X IS DIFFERENT
                        elif ((x_diff < 7) and (x_diff >= 3))  \
                                and ((y_diff < 2) and ((wf_y > 0 and node_y > 0) or (wf_y < 0 and node_y < 0))) \
                                and ((z_diff < 2) and ((wf_z > 0 and node_z > 0) or (wf_z < 0 and node_z < 0))):
                            wf_lower.append(node)

                # FROM CONNECTED POINTS CREATE COMBINATIONS FOR FACES
                wf_lower_combinations = list(itertools.combinations(wf_lower, 3))

                logging.debug('    wf_lower_combinations length: %s', len(wf_lower_combinations))
                logging.debug('    Data review: wf_lower_combinations %s', wf_lower_combinations)

                for combination in wf_lower_combinations:
                    node_1_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['x'])
                    node_2_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['x'])
                    node_3_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['x'])

                    node_1_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['y'])
                    node_2_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['y'])
                    node_3_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['y'])

                    node_1_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['z'])
                    node_2_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['z'])
                    node_3_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['z'])

                    if ((abs(node_1_x - node_2_x) < 1) and abs((node_2_x - node_3_x) < 1)) and not ((abs(node_1_y - node_2_y) < 1) and abs((node_2_y - node_3_y) < 1)):
                        wf_lower_tri.append(combination)
                    elif ((abs(node_1_y - node_2_y) < 1) and abs((node_2_y - node_3_y) < 1)) and not ((abs(node_1_x - node_2_x) < 1) and abs((node_2_x - node_3_x) < 1)):
                        wf_lower_tri.append(combination)

            logging.info('    wf_lower_tri length: %s', len(wf_lower_tri))
            logging.debug('    Data review: wf_lower_tri %s', wf_lower_tri)

            ####################################################################################################################
            # BUILD FRONT FACE POINT CLOUD
            workpiece_face_front_nodes = []
            workpiece_face_front_nodes.append(workpiece_front_face_nodes['node'].tolist())

            logging.info('    workpiece_face_front_nodes[0] length: %s', len(workpiece_face_front_nodes[0]))
            logging.debug('    Data review: workpiece_face_front_nodes[0] %s', workpiece_face_front_nodes[0])

            # BUILD THE FRONT FACE CONNECTED POINTS FOR TRIANGLES
            wf_front_tri = []
            for wf_node in workpiece_face_front_nodes[0]:
                wf_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['x'])
                wf_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['y'])
                wf_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['z'])
                wf_front = []
                wf_front.append(wf_node)

                for node in workpiece_face_front_nodes[0]:
                    if (wf_node != node) and node not in wf_front:
                        node_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['x'])
                        node_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['y'])
                        node_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['z'])

                        x_diff = abs(wf_x - node_x)
                        y_diff = abs(wf_y - node_y)
                        z_diff = abs(wf_z - node_z)

                        # THIS IS FOR FRONT AND BACK
                        # THESE ARE DEFINED BY X LOCATION
                        # IF X AND Z ARE ON THE SAME LINE AND Y IS DIFFERENT
                        if (x_diff < 3)  \
                                and (((y_diff < 4) and (y_diff > 2)) and ((wf_y > 0 and node_y > 0) or (wf_y < 0 and node_y < 0) or ((wf_y + node_y) < 1))) \
                                and ((z_diff < 2) and ((wf_z > 0 and node_z > 0) or (wf_z < 0 and node_z < 0))):
                            wf_front.append(node)
                        # IF X AND Y ARE ON THE SAME LINE AND Z IS DIFFERENT
                        elif (x_diff < 3) \
                                and ((y_diff < 2) and ((wf_y > 0 and node_y > 0) or (wf_y < 0 and node_y < 0))) \
                                and (((z_diff < 4) and (z_diff > 2)) and ((wf_z > 0 and node_z > 0) or (wf_z < 0 and node_z < 0) or ((wf_z + node_z) < 1))):
                            wf_front.append(node)

                # FROM CONNECTED POINTS CREATE COMBINATIONS FOR FACES
                wf_front_combinations = list(itertools.combinations(wf_front, 3))

                logging.debug('    wf_front_combinations length: %s', len(wf_front_combinations))
                logging.debug('    Data review: wf_front_combinations %s', wf_front_combinations)

                for combination in wf_front_combinations:
                    node_1_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['x'])
                    node_2_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['x'])
                    node_3_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['x'])

                    node_1_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['y'])
                    node_2_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['y'])
                    node_3_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['y'])

                    node_1_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['z'])
                    node_2_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['z'])
                    node_3_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['z'])

                    if ((abs(node_1_y - node_2_y) < 1) and abs((node_2_y - node_3_y) < 1)) and not ((abs(node_1_z - node_2_z) < 1) and abs((node_2_z - node_3_z) < 1)):
                        wf_front_tri.append(combination)
                    elif ((abs(node_1_z - node_2_z) < 1) and abs((node_2_z - node_3_z) < 1)) and not ((abs(node_1_y - node_2_y) < 1) and abs((node_2_y - node_3_y) < 1)):
                        wf_front_tri.append(combination)

            logging.info('    wf_front_tri length: %s', len(wf_front_tri))
            logging.debug('    Data review: wf_front_tri %s', wf_front_tri)


            if display_graphs == 1:
                wf_front_vertices = []
                for node in workpiece_face_front_nodes[0]:
                    vertex = []
                    node_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['x'])
                    node_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['y'])
                    node_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['z'])

                    vertex.append(node_x)
                    vertex.append(node_y)
                    vertex.append(node_z)

                    wf_front_vertices.append(vertex)

                workpiece_face_nodes = workpiece_face_front_nodes[0]

                workpiece_front_faces = []
                for tri in wf_front_tri:
                    new_tri = []
                    for node in tri:
                        node_locator = workpiece_face_nodes.index(node)
                        new_tri.append(node_locator)
                    workpiece_front_faces.append(new_tri)

                # DISPLAY POINT CLOUD
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(wf_front_vertices)
                o3d.visualization.draw_geometries([pcd], window_name="Workpiece Front Point Cloud")

                # DISPLAY MESH AND TRIANGULAR FACES
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(wf_front_vertices)
                mesh.triangles = o3d.utility.Vector3iVector(workpiece_front_faces)
                o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, window_name="Workpiece Front Mesh")

                # DISPLAY LINESET - VISUALIZATION OF MESH WITHOUT FACES
                lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
                o3d.visualization.draw_geometries([lineset], window_name="Workpiece Front Lineset")

            ####################################################################################################################
            # BUILD BACK FACE POINT CLOUD
            workpiece_face_back_nodes = []
            workpiece_face_back_nodes.append(workpiece_back_face_nodes['node'].tolist())

            logging.info('    workpiece_face_back_nodes[0] length: %s', len(workpiece_face_back_nodes[0]))
            logging.debug('    Data review: workpiece_face_back_nodes[0] %s', workpiece_face_back_nodes[0])

            # BUILD THE BACK FACE CONNECTED POINTS FOR TRIANGLES
            wf_back_tri = []
            for wf_node in workpiece_face_back_nodes[0]:
                wf_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['x'])
                wf_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['y'])
                wf_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['z'])
                wf_back = []
                wf_back.append(wf_node)

                for node in workpiece_face_back_nodes[0]:
                    if (wf_node != node) and node not in wf_back:
                        node_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['x'])
                        node_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['y'])
                        node_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['z'])

                        x_diff = abs(wf_x - node_x)
                        y_diff = abs(wf_y - node_y)
                        z_diff = abs(wf_z - node_z)

                        # THIS IS FOR FRONT AND BACK
                        # THESE ARE DEFINED BY X LOCATION
                        # IF X AND Z ARE ON THE SAME LINE AND Y IS DIFFERENT
                        if (x_diff < 3)  \
                                and (((y_diff < 4) and (y_diff > 2)) and ((wf_y > 0 and node_y > 0) or (wf_y < 0 and node_y < 0) or ((wf_y + node_y) < 1))) \
                                and ((z_diff < 2) and ((wf_z > 0 and node_z > 0) or (wf_z < 0 and node_z < 0))):
                            wf_back.append(node)
                        # IF X AND Y ARE ON THE SAME LINE AND Z IS DIFFERENT
                        elif (x_diff < 3) \
                                and ((y_diff < 2) and ((wf_y > 0 and node_y > 0) or (wf_y < 0 and node_y < 0))) \
                                and (((z_diff < 4) and (z_diff > 2)) and ((wf_z > 0 and node_z > 0) or (wf_z < 0 and node_z < 0) or ((wf_z + node_z) < 1))):
                            wf_back.append(node)

                # FROM CONNECTED POINTS CREATE COMBINATIONS FOR FACES
                wf_back_combinations = list(itertools.combinations(wf_back, 3))

                logging.debug('    wf_back_combinations length: %s', len(wf_back_combinations))
                logging.debug('    Data review: wf_back_combinations %s', wf_back_combinations)

                for combination in wf_back_combinations:
                    node_1_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['x'])
                    node_2_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['x'])
                    node_3_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['x'])

                    node_1_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['y'])
                    node_2_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['y'])
                    node_3_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['y'])

                    node_1_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['z'])
                    node_2_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['z'])
                    node_3_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['z'])

                    if ((abs(node_1_y - node_2_y) < 1) and abs((node_2_y - node_3_y) < 1)) and not ((abs(node_1_z - node_2_z) < 1) and abs((node_2_z - node_3_z) < 1)):
                        wf_back_tri.append(combination)
                    elif ((abs(node_1_z - node_2_z) < 1) and abs((node_2_z - node_3_z) < 1)) and not ((abs(node_1_y - node_2_y) < 1) and abs((node_2_y - node_3_y) < 1)):
                        wf_back_tri.append(combination)

            logging.info('    wf_back_tri length: %s', len(wf_back_tri))
            logging.debug('    Data review: wf_back_tri %s', wf_back_tri)

            ####################################################################################################################
            # BUILD RIGHT FACE POINT CLOUD
            workpiece_face_right_nodes = []
            workpiece_face_right_nodes.append(workpiece_right_face_nodes['node'].tolist())

            logging.info('    workpiece_face_right_nodes[0] length: %s', len(workpiece_face_right_nodes[0]))
            logging.debug('    Data review: workpiece_face_right_nodes[0] %s', workpiece_face_right_nodes[0])

            # BUILD THE RIGHT FACE CONNECTED POINTS FOR TRIANGLES
            wf_right_tri = []
            for wf_node in workpiece_face_right_nodes[0]:
                wf_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['x'])
                wf_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['y'])
                wf_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['z'])
                wf_right = []
                wf_right.append(wf_node)

                for node in workpiece_face_right_nodes[0]:
                    if (wf_node != node) and node not in wf_right:
                        node_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['x'])
                        node_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['y'])
                        node_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['z'])

                        x_diff = abs(wf_x - node_x)
                        y_diff = abs(wf_y - node_y)
                        z_diff = abs(wf_z - node_z)

                        # THIS IS FOR RIGHT AND LEFT
                        # THESE ARE DEFINED BY Y LOCATION
                        # IF Y AND Z ARE ON THE SAME LINE AND Z IS DIFFERENT
                        if ((x_diff < 7) and (x_diff >= 3))  \
                                and ((y_diff < 2) and ((wf_y > 0 and node_y > 0) or (wf_y < 0 and node_y < 0))) \
                                and ((z_diff < 2) and ((wf_z > 0 and node_z > 0) or (wf_z < 0 and node_z < 0))):
                            wf_right.append(node)
                        # IF X AND Y ARE ON THE SAME LINE AND X IS DIFFERENT
                        elif (x_diff < 3) \
                                and ((y_diff < 2) and ((wf_y > 0 and node_y > 0) or (wf_y < 0 and node_y < 0))) \
                                and ((z_diff < 4) and (z_diff > 2) and ((wf_z > 0 and node_z > 0) or (wf_z < 0 and node_z < 0) or ((wf_z + node_z) < 1))):
                            wf_right.append(node)

                # FROM CONNECTED POINTS CREATE COMBINATIONS FOR FACES
                wf_right_combinations = list(itertools.combinations(wf_right, 3))

                logging.debug('    wf_right_combinations length: %s', len(wf_right_combinations))
                logging.debug('    Data review: wf_right_combinations %s', wf_right_combinations)

                for combination in wf_right_combinations:
                    node_1_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['x'])
                    node_2_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['x'])
                    node_3_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['x'])

                    node_1_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['y'])
                    node_2_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['y'])
                    node_3_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['y'])

                    node_1_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['z'])
                    node_2_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['z'])
                    node_3_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['z'])

                    if ((abs(node_1_x - node_2_x) < 1) and abs((node_2_x - node_3_x) < 1)) and not ((abs(node_1_z - node_2_z) < 1) and abs((node_2_z - node_3_z) < 1)):
                        wf_right_tri.append(combination)
                    elif ((abs(node_1_z - node_2_z) < 1) and abs((node_2_z - node_3_z) < 1)) and not ((abs(node_1_x - node_2_x) < 1) and abs((node_2_x - node_3_x) < 1)):
                        wf_right_tri.append(combination)

            logging.info('    wf_right_tri length: %s', len(wf_right_tri))
            logging.debug('    Data review: wf_right_tri %s', wf_right_tri)

            if display_graphs == 1:
                wf_right_vertices = []
                for node in workpiece_face_right_nodes[0]:
                    vertex = []
                    node_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['x'])
                    node_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['y'])
                    node_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['z'])

                    vertex.append(node_x)
                    vertex.append(node_y)
                    vertex.append(node_z)

                    wf_right_vertices.append(vertex)

                workpiece_face_nodes = workpiece_face_right_nodes[0]

                workpiece_right_faces = []
                for tri in wf_right_tri:
                    new_tri = []
                    for node in tri:
                        node_locator = workpiece_face_nodes.index(node)
                        new_tri.append(node_locator)
                    workpiece_right_faces.append(new_tri)

                # DISPLAY POINT CLOUD
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(wf_right_vertices)
                o3d.visualization.draw_geometries([pcd], window_name="Workpiece Right Point Cloud")

                # DISPLAY MESH AND TRIANGULAR FACES
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(wf_right_vertices)
                mesh.triangles = o3d.utility.Vector3iVector(workpiece_right_faces)
                o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, window_name="Workpiece Right Mesh")

                # DISPLAY LINESET - VISUALIZATION OF MESH WITHOUT FACES
                lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
                o3d.visualization.draw_geometries([lineset], window_name="Workpiece Right Lineset")

            ####################################################################################################################
            # BUILD LEFT FACE POINT CLOUD
            workpiece_face_left_nodes = []
            workpiece_face_left_nodes.append(workpiece_left_face_nodes['node'].tolist())

            logging.info('    workpiece_face_left_nodes[0] length: %s', len(workpiece_face_left_nodes[0]))
            logging.debug('    Data review: workpiece_face_left_nodes[0] %s', workpiece_face_left_nodes[0])

            # BUILD THE LEFT FACE CONNECTED POINTS FOR TRIANGLES
            wf_left_tri = []
            for wf_node in workpiece_face_left_nodes[0]:
                wf_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['x'])
                wf_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['y'])
                wf_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wf_node]['z'])
                wf_left = []
                wf_left.append(wf_node)

                for node in workpiece_face_left_nodes[0]:
                    if (wf_node != node) and node not in wf_left:
                        node_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['x'])
                        node_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['y'])
                        node_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['z'])

                        x_diff = abs(wf_x - node_x)
                        y_diff = abs(wf_y - node_y)
                        z_diff = abs(wf_z - node_z)

                        # THIS IS FOR RIGHT AND LEFT
                        # THESE ARE DEFINED BY Y LOCATION
                        # IF Y AND Z ARE ON THE SAME LINE AND Z IS DIFFERENT
                        if ((x_diff < 7) and (x_diff >= 3))  \
                                and ((y_diff < 2) and ((wf_y > 0 and node_y > 0) or (wf_y < 0 and node_y < 0))) \
                                and ((z_diff < 2) and ((wf_z > 0 and node_z > 0) or (wf_z < 0 and node_z < 0))):
                            wf_left.append(node)
                        # IF X AND Y ARE ON THE SAME LINE AND X IS DIFFERENT
                        elif (x_diff < 3) \
                                and ((y_diff < 2) and ((wf_y > 0 and node_y > 0) or (wf_y < 0 and node_y < 0))) \
                                and ((z_diff < 4) and (z_diff > 2) and ((wf_z > 0 and node_z > 0) or (wf_z < 0 and node_z < 0) or ((wf_z + node_z) < 1))):
                            wf_left.append(node)

                # FROM CONNECTED POINTS CREATE COMBINATIONS FOR FACES
                wf_left_combinations = list(itertools.combinations(wf_left, 3))

                logging.debug('    wf_left_combinations length: %s', len(wf_left_combinations))
                logging.debug('    Data review: wf_left_combinations %s', wf_left_combinations)

                for combination in wf_left_combinations:
                    node_1_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['x'])
                    node_2_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['x'])
                    node_3_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['x'])

                    node_1_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['y'])
                    node_2_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['y'])
                    node_3_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['y'])

                    node_1_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[0]]['z'])
                    node_2_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[1]]['z'])
                    node_3_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == combination[2]]['z'])

                    if ((abs(node_1_x - node_2_x) < 1) and abs((node_2_x - node_3_x) < 1)) and not ((abs(node_1_z - node_2_z) < 1) and abs((node_2_z - node_3_z) < 1)):
                        wf_left_tri.append(combination)
                    elif ((abs(node_1_z - node_2_z) < 1) and abs((node_2_z - node_3_z) < 1)) and not ((abs(node_1_x - node_2_x) < 1) and abs((node_2_x - node_3_x) < 1)):
                        wf_left_tri.append(combination)

            logging.info('    wf_left_tri length: %s', len(wf_left_tri))
            logging.debug('    Data review: wf_left_tri %s', wf_left_tri)

            ####################################################################################################################
            # BUILD THE OVERALL WORKPIECE VERTICES LIST
            workpiece_vertices = []
            workpiece_face_nodes = workpiece_face_upper_nodes[0] + workpiece_face_lower_nodes[0] + workpiece_face_front_nodes[0] + workpiece_face_back_nodes[0] + workpiece_face_right_nodes[0] + workpiece_face_left_nodes[0]

            logging.info('    workpiece_face_nodes length: %s', len(workpiece_face_nodes))
            logging.debug('    Data review: workpiece_face_nodes %s', workpiece_face_nodes)

            for node in workpiece_face_nodes:
                vertex = []
                node_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['x'])
                node_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['y'])
                node_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == node]['z'])

                vertex.append(node_x)
                vertex.append(node_y)
                vertex.append(node_z)

                workpiece_vertices.append(vertex)

            logging.info('    workpiece_vertices length: %s', len(workpiece_vertices))
            logging.debug('    Data review: workpiece_vertices %s', workpiece_vertices)

            workpiece_face_nodes = []
            workpiece_face_nodes.append(workpiece_upper_face_nodes['node'].tolist())
            workpiece_face_nodes.append(workpiece_lower_face_nodes['node'].tolist())
            workpiece_face_nodes.append(workpiece_front_face_nodes['node'].tolist())
            workpiece_face_nodes.append(workpiece_back_face_nodes['node'].tolist())
            workpiece_face_nodes.append(workpiece_right_face_nodes['node'].tolist())
            workpiece_face_nodes.append(workpiece_left_face_nodes['node'].tolist())

            workpiece_face_nodes = list(itertools.chain.from_iterable(workpiece_face_nodes))

            upper_workpiece_face_nodes = []
            upper_workpiece_face_nodes.append(workpiece_upper_face_nodes['node'].tolist())
            upper_workpiece_face_nodes = list(itertools.chain.from_iterable(upper_workpiece_face_nodes))
            lower_workpiece_face_nodes = []
            lower_workpiece_face_nodes.append(workpiece_lower_face_nodes['node'].tolist())
            lower_workpiece_face_nodes = list(itertools.chain.from_iterable(lower_workpiece_face_nodes))

            logging.info('    post-itertools workpiece_face_nodes length: %s', len(workpiece_face_nodes))
            logging.debug('    Data review: post-itertools workpiece_face_nodes %s', workpiece_face_nodes)

            wf_tri = wf_upper_tri + wf_lower_tri + wf_front_tri + wf_back_tri + wf_right_tri + wf_left_tri

            logging.info('    wf_tri length: %s', len(wf_tri))
            logging.debug('    Data review: wf_tri %s', wf_tri)

            # BUILD THE WORKPIECE FACES REFERENCING THE NODES FROM VERTICES
            workpiece_faces = []
            for tri in wf_tri:
                new_tri = []
                for node in tri:
                    node_locator = workpiece_face_nodes.index(node)
                    new_tri.append(node_locator)
                workpiece_faces.append(new_tri)

            logging.info('    workpiece_faces length: %s', len(workpiece_faces))
            logging.debug('    Data review: workpiece_faces %s', workpiece_faces)

            if display_graphs == 1:
                # DISPLAY POINT CLOUD
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(workpiece_vertices)
                o3d.visualization.draw_geometries([pcd], window_name="Workpiece Point Cloud")

                # DISPLAY MESH AND TRIANGULAR FACES
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(workpiece_vertices)
                mesh.triangles = o3d.utility.Vector3iVector(workpiece_faces)
                o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, window_name="Workpiece Mesh")

                # DISPLAY LINESET - VISUALIZATION OF MESH WITHOUT FACES
                lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
                o3d.visualization.draw_geometries([lineset], window_name="Workpiece Lineset")

            # CREATE THE WORKPIECE SURFACE FACES CSV
            workpiece_surface_faces_df = pd.DataFrame(workpiece_faces)
            workpiece_surface_faces_df.columns = ['node_1', 'node_2', 'node_3']
            logging.info('Saving surface faces to: %s', workpiece_surface_faces_csv_destination)
            workpiece_surface_faces_df.to_csv(workpiece_surface_faces_csv_destination)

            ####################################################################################################################
            # BUILD TOOL POINT CLOUD AND MESH
            ####################################################################################################################
            tool_node_details_df = graph_df.loc[graph_df['ID'] == 2]
            upper_tool_node_details_df = tool_node_details_df.loc[tool_node_details_df['z'] > 0]
            lower_tool_node_details_df = tool_node_details_df.loc[tool_node_details_df['z'] < 0]

            logging.info('    upper_tool_node_details_df shape: %s', upper_tool_node_details_df.shape)
            logging.debug('    Data review: upper_tool_node_details_df %s', upper_tool_node_details_df)
            logging.info('    lower_tool_node_details_df shape: %s', lower_tool_node_details_df.shape)
            logging.debug('    Data review: lower_tool_node_details_df %s', lower_tool_node_details_df)
            logging.info('    tool_node_details_df shape: %s', tool_node_details_df.shape)
            logging.debug('    Data review: tool_node_details_df %s', tool_node_details_df)

            tool_nodes = []
            tool_nodes.append(tool_node_details_df['node'].tolist())
            upper_tool_nodes = []
            upper_tool_nodes.append(upper_tool_node_details_df['node'].tolist())
            lower_tool_nodes = []
            lower_tool_nodes.append(lower_tool_node_details_df['node'].tolist())


            logging.info('    upper_tool_nodes[0] length: %s', len(upper_tool_nodes[0]))
            logging.debug('    Data review: upper_tool_nodes[0] %s', upper_tool_nodes[0])
            logging.info('    lower_tool_nodes[0] length: %s', len(lower_tool_nodes[0]))
            logging.debug('    Data review: lower_tool_nodes[0] %s', lower_tool_nodes[0])
            logging.info('    tool_nodes[0] length: %s', len(tool_nodes[0]))
            logging.debug('    Data review: tool_nodes[0] %s', tool_nodes[0])

            for index, row in tool_node_details_df.iterrows():
                node_1 = row['node']
                node_1_x = row['x']
                node_1_y = row['y']
                node_1_z = row['z']
                node_1_type = row['ID']
                sim_nodes.append(node_1)
                sim_node_attributes.append([node_1_x, node_1_y, node_1_z, node_1_type])

                neighbor_list = row['neighbors']

                neighbor_list = neighbor_list.replace("[", "")
                neighbor_list = neighbor_list.replace("]", "")
                neighbor_list = neighbor_list.replace(" ", "")

                neighbor_list = neighbor_list.split(',')

                for node_2 in neighbor_list:
                    edge_nodes = [int(node_1), int(node_2)]
                    sorted_edge_nodes = sorted(edge_nodes)

                    if sorted_edge_nodes not in edges:
                        if sorted_edge_nodes not in sim_edges:
                            sim_edges.append(sorted_edge_nodes)

            edges.append([simulation_counter, sim_edges])
            nodes.append([simulation_counter, sim_nodes])
            node_attributes.append([simulation_counter, sim_node_attributes])

            mesh_pos = []
            node_type = []

            for node in nodes[0][1]:
                node_index = nodes[0][1].index(node)
                mesh_pos.append(node_attributes[0][1][node_index][:3])
                node_type.append(node_attributes[0][1][node_index][3:])

            # DISPLAY THE 'cells' COMPONENT; THIS IS CONSISTENT ACROSS STEPS IN A SIMULATION
            tool_tri = generate_faces(nodes, edges, node_attributes)

            # BUILD THE TOOL FACES REFERENCING THE NODES FROM VERTICES
            upper_tool_nodes = upper_tool_nodes[0]
            tool_nodes = tool_nodes[0]
            tool_faces = []
            for tri in tool_tri:
                new_tri = []
                for node in tri:
                    node_locator = tool_nodes.index(node)
                    new_tri.append(node_locator)
                tool_faces.append(new_tri)

            logging.info('    tool_faces length: %s', len(tool_faces))
            logging.debug('    Data review: tool_faces %s', tool_faces)

            upper_tool_vertices = []
            for node in upper_tool_nodes:
                vertex = []
                node_x = float(upper_tool_node_details_df.loc[upper_tool_node_details_df['node'] == node]['x'])
                node_y = float(upper_tool_node_details_df.loc[upper_tool_node_details_df['node'] == node]['y'])
                node_z = float(upper_tool_node_details_df.loc[upper_tool_node_details_df['node'] == node]['z'])

                vertex.append(node_x)
                vertex.append(node_y)
                vertex.append(node_z)

                upper_tool_vertices.append(vertex)

            lower_tool_nodes = lower_tool_nodes[0]

            lower_tool_vertices = []
            for node in lower_tool_nodes:
                vertex = []
                node_x = float(lower_tool_node_details_df.loc[lower_tool_node_details_df['node'] == node]['x'])
                node_y = float(lower_tool_node_details_df.loc[lower_tool_node_details_df['node'] == node]['y'])
                node_z = float(lower_tool_node_details_df.loc[lower_tool_node_details_df['node'] == node]['z'])

                vertex.append(node_x)
                vertex.append(node_y)
                vertex.append(node_z)

                lower_tool_vertices.append(vertex)

            tool_vertices = upper_tool_vertices + lower_tool_vertices
            logging.info('    tool_vertices length: %s', len(tool_vertices))
            logging.debug('    Data review: tool_vertices %s', tool_vertices)

            if display_graphs == 1:
                # DISPLAY POINT CLOUD
                tool_pcd = o3d.geometry.PointCloud()
                tool_pcd.points = o3d.utility.Vector3dVector(tool_vertices)
                o3d.visualization.draw_geometries([tool_pcd], window_name="Tool Point Cloud")

                # DISPLAY MESH AND TRIANGULAR FACES
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(tool_vertices)
                mesh.triangles = o3d.utility.Vector3iVector(tool_faces)
                o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True, window_name="Tool Mesh")

                # DISPLAY LINESET - VISUALIZATION OF MESH WITHOUT FACES
                lineset = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
                o3d.visualization.draw_geometries([lineset], window_name="Tool Lineset")

            all_vertices = workpiece_vertices + tool_vertices
            all_nodes = workpiece_face_nodes + tool_nodes


            logging.info('    all_vertices length: %s', len(all_vertices))
            logging.debug('    Data review: all_vertices %s', all_vertices)
            logging.info('    all_nodes length: %s', len(all_nodes))
            logging.debug('    Data review: all_nodes %s', all_nodes)

            # CREATE THE ALL VERTICES CSV
            workpiece_vertices_df = pd.DataFrame(workpiece_vertices)
            workpiece_vertices_df.columns = ['x', 'y', 'z']
            workpiece_vertices_df['ID'] = 1

            tool_vertices_df = pd.DataFrame(tool_vertices)
            tool_vertices_df.columns = ['x', 'y', 'z']
            tool_vertices_df['ID'] = 2

            all_vertices_df = pd.concat([workpiece_vertices_df, tool_vertices_df])
            all_vertices_df = all_vertices_df.reset_index(drop=True)

            logging.info('    all_vertices_df shape: %s', all_vertices_df.shape)
            logging.debug('    Data review: all_vertices_df %s', all_vertices_df)

            logging.info('Saving all vertices to: %s', all_vertices_csv_destination)
            all_vertices_df.to_csv(all_vertices_csv_destination)


            # BUILD THE COMPLETE FACES REFERENCING THE NODES FROM VERTICES
            all_faces = []
            for tri in tool_tri:
                new_tri = []
                for node in tri:
                    node_locator = all_nodes.index(node)
                    new_tri.append(node_locator)
                tool_faces.append(new_tri)

            logging.info('    all_faces length: %s', len(all_faces))
            logging.debug('    Data review: all_faces %s', all_faces)

            # CREATE THE SURFACE FACES CSV
            tool_surface_faces_df = pd.DataFrame(tool_faces)
            tool_surface_faces_df.columns = ['node_1', 'node_2', 'node_3']
            logging.info('Saving tool surface faces to: %s', tool_surface_faces_csv_destination)
            tool_surface_faces_df.to_csv(tool_surface_faces_csv_destination)


            if display_graphs == 1:
                # DISPLAY COMPLETE POINT CLOUD
                complete_pcd = o3d.geometry.PointCloud()
                complete_pcd.points = o3d.utility.Vector3dVector(all_vertices)
                o3d.visualization.draw_geometries([complete_pcd], window_name="Complete Point Cloud")

                # DISPLAY MESH AND TRIANGULAR FACES
                complete_mesh = o3d.geometry.TriangleMesh()
                complete_mesh.vertices = o3d.utility.Vector3dVector(all_vertices)
                complete_mesh.triangles = o3d.utility.Vector3iVector(all_faces)
                o3d.visualization.draw_geometries([complete_mesh], mesh_show_wireframe=True, window_name="Complete Mesh")

                # DISPLAY LINESET - VISUALIZATION OF MESH WITHOUT FACES
                lineset = o3d.geometry.LineSet.create_from_triangle_mesh(complete_mesh)
                o3d.visualization.draw_geometries([lineset], window_name="Complete Lineset")



            build_connections = 1
            if build_connections == 1:

                if create_tool_to_wp == 1:
                    # BUILD CONNECTION BETWEEN TOOL AND WORKPIECE
                    # ITERATE THROUGH THE UPPER TOOL NODES TO FIND THE N CLOSEST WORKPIECE NODES
                    logging.info('... Building connection between tool and workpiece\n                              ... this takes some time.')

                    tool_to_wp_tri = []

                    for tool_node in upper_tool_nodes:
                        neighbor_list = []
                        neighbor_node_list = []
                        distance_list = []

                        tool_node_x = float(upper_tool_node_details_df.loc[upper_tool_node_details_df['node'] == tool_node]['x'])
                        tool_node_y = float(upper_tool_node_details_df.loc[upper_tool_node_details_df['node'] == tool_node]['y'])
                        tool_node_z = float(upper_tool_node_details_df.loc[upper_tool_node_details_df['node'] == tool_node]['z'])
                        P = [tool_node_x, tool_node_y, tool_node_z]

                        for wp_node in workpiece_face_nodes:
                            wp_node_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wp_node]['x'])
                            wp_node_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wp_node]['y'])
                            wp_node_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wp_node]['z'])
                            Q = [wp_node_x, wp_node_y, wp_node_z]

                            euclidian_dist = math.dist(P, Q)

                            if wp_node not in neighbor_node_list:
                                distance_list.append(euclidian_dist)
                                neighbor_node_list.append(wp_node)

                        # GET THE SORTED DISTANCES AND RETURN THE n (tool_to_wp_connections) CLOSEST NODE INDEXES
                        sorted_distances = sorted(enumerate(distance_list), key=lambda x: x[1])[:tool_to_wp_connections]
                        sorted_index, sorted_value = map(list, zip(*sorted_distances))

                        for sorted_idx in sorted_index:
                            neighbor_list.append(neighbor_node_list[sorted_idx])

                        # CREATE TRIANGLES BASED ON DISTANCE
                        # THIS IS NOT INTENDED TO BE A FACE MESH IT IS SIMPLY PROVIDING EDGE INFORMATION
                        neighbor_face = []
                        neighbor_face.append(tool_node)
                        for neighbor in neighbor_list:
                            neighbor_face.append(neighbor)
                            if len(neighbor_face) == 3:
                                tool_to_wp_tri.append(neighbor_face)
                                neighbor_face = []
                                neighbor_face.append(tool_node)

                    # ITERATE THROUGH THE LOWER TOOL NODES TO FIND THE N CLOSEST WORKPIECE NODES
                    for tool_node in lower_tool_nodes:
                        neighbor_list = []
                        neighbor_node_list = []
                        distance_list = []

                        tool_node_x = float(lower_tool_node_details_df.loc[lower_tool_node_details_df['node'] == tool_node]['x'])
                        tool_node_y = float(lower_tool_node_details_df.loc[lower_tool_node_details_df['node'] == tool_node]['y'])
                        tool_node_z = float(lower_tool_node_details_df.loc[lower_tool_node_details_df['node'] == tool_node]['z'])
                        P = [tool_node_x, tool_node_y, tool_node_z]

                        for wp_node in workpiece_face_nodes:
                            wp_node_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wp_node]['x'])
                            wp_node_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wp_node]['y'])
                            wp_node_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wp_node]['z'])
                            Q = [wp_node_x, wp_node_y, wp_node_z]

                            euclidian_dist = math.dist(P, Q)

                            if wp_node not in neighbor_node_list:
                                distance_list.append(euclidian_dist)
                                neighbor_node_list.append(wp_node)

                        # GET THE SORTED DISTANCES AND RETURN THE 10 CLOSEST NODE INDEXES
                        sorted_distances = sorted(enumerate(distance_list), key=lambda x: x[1])[:tool_to_wp_connections]
                        sorted_index, sorted_value = map(list, zip(*sorted_distances))

                        for sorted_idx in sorted_index:
                            neighbor_list.append(neighbor_node_list[sorted_idx])

                        # CREATE TRIANGLES BASED ON DISTANCE
                        # THIS IS NOT INTENDED TO BE A FACE MESH IT IS SIMPLY PROVIDING EDGE INFORMATION
                        neighbor_face = []
                        neighbor_face.append(tool_node)
                        for neighbor in neighbor_list:
                            neighbor_face.append(neighbor)
                            if len(neighbor_face) == 3:
                                tool_to_wp_tri.append(neighbor_face)
                                neighbor_face = []
                                neighbor_face.append(tool_node)


                    # BUILD THE COMPLETE FACES REFERENCING THE NODES FROM VERTICES
                    tool_to_wp_faces = []
                    for tri in tool_to_wp_tri:
                        new_tri = []
                        for node in tri:
                            node_locator = all_nodes.index(node)
                            new_tri.append(node_locator)
                        tool_to_wp_faces.append(new_tri)


                    # CREATE THE CONNECTION FACES CSV
                    tool_to_wp_faces_df = pd.DataFrame(tool_to_wp_faces)
                    tool_to_wp_faces_df.columns = ['node_1', 'node_2', 'node_3']

                    logging.debug('Saving tool to workpiece world connection faces to: %s', tool_to_wp_connection_faces_csv_destination)
                    tool_to_wp_faces_df.to_csv(tool_to_wp_connection_faces_csv_destination)




                if create_wp_to_tool == 1:
                    # BUILD CONNECTION BETWEEN WORKPIECE AND TOOL
                    # ITERATE THROUGH THE UPPER WORKPIECE NODES TO FIND THE N CLOSEST TOOL NODES
                    logging.info('... Building connection between workpiece and tool\n                              ... this takes some time.')

                    wp_to_tool_tri = []

                    for wp_node in upper_workpiece_face_nodes:
                        neighbor_list = []
                        neighbor_node_list = []
                        distance_list = []

                        wp_node_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wp_node]['x'])
                        wp_node_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wp_node]['y'])
                        wp_node_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wp_node]['z'])
                        P = [wp_node_x, wp_node_y, wp_node_z]

                        for tool_node in upper_tool_nodes:
                            tool_node_x = float(upper_tool_node_details_df.loc[upper_tool_node_details_df['node'] == tool_node]['x'])
                            tool_node_y = float(upper_tool_node_details_df.loc[upper_tool_node_details_df['node'] == tool_node]['y'])
                            tool_node_z = float(upper_tool_node_details_df.loc[upper_tool_node_details_df['node'] == tool_node]['z'])
                            Q = [tool_node_x, tool_node_y, tool_node_z]

                            euclidian_dist = math.dist(P, Q)

                            if tool_node not in neighbor_node_list:
                                distance_list.append(euclidian_dist)
                                neighbor_node_list.append(tool_node)

                        # GET THE SORTED DISTANCES AND RETURN THE N CLOSEST NODE INDEXES
                        sorted_distances = sorted(enumerate(distance_list), key=lambda x: x[1])[:tool_to_wp_connections]
                        sorted_index, sorted_value = map(list, zip(*sorted_distances))

                        for sorted_idx in sorted_index:
                            neighbor_list.append(neighbor_node_list[sorted_idx])

                        # CREATE TRIANGLES BASED ON DISTANCE
                        # THIS IS NOT INTENDED TO BE A FACE MESH IT IS SIMPLY PROVIDING EDGE INFORMATION
                        neighbor_face = []
                        neighbor_face.append(wp_node)
                        for neighbor in neighbor_list:
                            neighbor_face.append(neighbor)
                            if len(neighbor_face) == 3:
                                wp_to_tool_tri.append(neighbor_face)
                                neighbor_face = []
                                neighbor_face.append(wp_node)

                    # ITERATE THROUGH THE LOWER FACE NODES TO FIND THE N CLOSEST TOOL NODES
                    for wp_node in lower_workpiece_face_nodes:
                        neighbor_list = []
                        neighbor_node_list = []
                        distance_list = []

                        wp_node_x = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wp_node]['x'])
                        wp_node_y = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wp_node]['y'])
                        wp_node_z = float(workpiece_node_details_df.loc[workpiece_node_details_df['node'] == wp_node]['z'])
                        P = [wp_node_x, wp_node_y, wp_node_z]

                        for tool_node in lower_tool_nodes:
                            tool_node_x = float(lower_tool_node_details_df.loc[lower_tool_node_details_df['node'] == tool_node]['x'])
                            tool_node_y = float(lower_tool_node_details_df.loc[lower_tool_node_details_df['node'] == tool_node]['y'])
                            tool_node_z = float(lower_tool_node_details_df.loc[lower_tool_node_details_df['node'] == tool_node]['z'])
                            Q = [tool_node_x, tool_node_y, tool_node_z]

                            euclidian_dist = math.dist(P, Q)

                            if tool_node not in neighbor_node_list:
                                distance_list.append(euclidian_dist)
                                neighbor_node_list.append(tool_node)

                        # GET THE SORTED DISTANCES AND RETURN THE N CLOSEST NODE INDEXES
                        sorted_distances = sorted(enumerate(distance_list), key=lambda x: x[1])[:tool_to_wp_connections]
                        sorted_index, sorted_value = map(list, zip(*sorted_distances))

                        for sorted_idx in sorted_index:
                            neighbor_list.append(neighbor_node_list[sorted_idx])

                        # CREATE TRIANGLES BASED ON DISTANCE
                        # THIS IS NOT INTENDED TO BE A FACE MESH IT IS SIMPLY PROVIDING EDGE INFORMATION
                        neighbor_face = []
                        neighbor_face.append(wp_node)
                        for neighbor in neighbor_list:
                            neighbor_face.append(neighbor)
                            if len(neighbor_face) == 3:
                                wp_to_tool_tri.append(neighbor_face)
                                neighbor_face = []
                                neighbor_face.append(wp_node)

                    # BUILD THE COMPLETE FACES REFERENCING THE NODES FROM VERTICES
                    wp_to_tool_faces = []
                    for tri in wp_to_tool_tri:
                        new_tri = []
                        for node in tri:
                            node_locator = all_nodes.index(node)
                            new_tri.append(node_locator)
                        wp_to_tool_faces.append(new_tri)

                    # CREATE THE CONNECTION FACES CSV
                    wp_to_tool_faces_df = pd.DataFrame(wp_to_tool_faces)
                    wp_to_tool_faces_df.columns = ['node_1', 'node_2', 'node_3']

                    logging.debug('Saving workpiece to tool world connection faces to: %s', wp_to_tool_connection_faces_csv_destination)
                    wp_to_tool_faces_df.to_csv(wp_to_tool_connection_faces_csv_destination)



    tool_to_wp_connections += 1