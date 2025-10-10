import logging
import os
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
import open3d as o3d
from itertools import chain

def triangles_to_edges(faces):
    """Computes mesh edges from triangles.
     Note that this triangles_to_edges method was provided as part of the
     code release for the MeshGraphNets paper by DeepMind, available here:
     https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets
    """
    # collect edges from triangles
    edges = tf.concat([faces[:, 0:2],
            faces[:, 1:3],
            tf.stack([faces[:, 2], faces[:, 0]], axis=1)], axis=0)
    # those edges are sometimes duplicated (within the mesh) and sometimes
    # single (at the mesh boundary).
    # sort & pack edges as single tf.int64
    receivers = tf.reduce_min(edges, axis=1)
    senders = tf.reduce_max(edges, axis=1)
    packed_edges = tf.bitcast(tf.stack([senders, receivers], axis=1), tf.int64)
    # remove duplicates and unpack
    unique_edges = tf.bitcast(tf.unique(packed_edges)[0], tf.int32)
    senders, receivers = tf.unstack(unique_edges, axis=1)
    # create two-way connectivity
    return (tf.concat([senders, receivers], axis=0), tf.concat([receivers, senders], axis=0))


def generate_face_array(face_details_df, node_array):
    face_counter = 0
    for index, face_row in face_details_df.iterrows():

        node_1_number = face_row['node_1']
        node_2_number = face_row['node_2']
        node_3_number = face_row['node_3']

        node_1_detail_row = np.where(node_array[:, 1] == node_1_number)
        node_1_index = node_1_detail_row[0][0]
        node_2_detail_row = np.where(node_array[:, 1] == node_2_number)
        node_2_index = node_2_detail_row[0][0]
        node_3_detail_row = np.where(node_array[:, 1] == node_3_number)
        node_3_index = node_3_detail_row[0][0]

        face_index_row = [node_1_index, node_2_index, node_3_index]

        if face_counter == 0:
            face_array = np.array(face_index_row)
        else:
            face_array = np.vstack([face_array, face_index_row])

        face_counter += 1

    return face_array

def simple_generate_edge_attributes(mesh_pos_array, edge_index):

    u_i = torch.tensor(mesh_pos_array)[edge_index[0]]
    u_j = torch.tensor(mesh_pos_array)[edge_index[1]]

    relative_world_pos = u_i - u_j
    relative_world_pos_norm = torch.linalg.vector_norm(relative_world_pos, ord=2, dim=1, keepdim=True)

    edge_attr = torch.cat((relative_world_pos, relative_world_pos_norm), dim=-1).type(torch.float)
    edge_attr = torch.reshape(edge_attr, (1, edge_attr.shape[0], edge_attr.shape[1]))

    return edge_attr

def generate_edge_attributes(mesh_pos_array, edge_index, type):

    u_i = torch.tensor(mesh_pos_array)[edge_index[0]]
    u_j = torch.tensor(mesh_pos_array)[edge_index[1]]

    relative_world_pos = u_i - u_j
    relative_world_pos_norm = torch.linalg.vector_norm(relative_world_pos, ord=2, dim=1, keepdim=True)

    if type == "mesh":
        relative_mesh_pos = relative_world_pos
        relative_mesh_pos_norm = relative_world_pos_norm
    elif type == "world":
        relative_mesh_pos = relative_world_pos.detach().clone()
        relative_mesh_pos.fill_(10)
        relative_mesh_pos_norm = torch.linalg.vector_norm(relative_mesh_pos, ord=2, dim=1, keepdim=True)

    edge_attr = torch.cat((relative_mesh_pos, relative_mesh_pos_norm, relative_world_pos, relative_world_pos_norm), dim=-1).type(torch.float)
    edge_attr = torch.reshape(edge_attr, (1, edge_attr.shape[0], edge_attr.shape[1]))


    return edge_attr

#######################################################################################################
# MAIN PROGRAM BEGINS HERE
#######################################################################################################

logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s:%(levelname)s: %(message)s')

generate_initial_tensors = 1
display_graphs = 0

# SET THE NUMBER OF EDGES FROM EACH TOOL NODE TO THE WORKPIECE
tool_to_wp_connections = 3
max_tool_to_wp_connections = 3

root_dir = "/home/FORGING/input/"
episode_file = root_dir + "graph_csv/episode_summary.csv"

episode_desc = pd.read_csv(episode_file)
logging.debug('Episode Description \n--------------------------------------\n %s\n', episode_desc)

simulation_counter = -1
simulation_columns = ['sim_counter', 'step_counter', 'node_file', 'face_file']

simulation_files = pd.DataFrame(columns=simulation_columns)
simulation_counter = -1


if generate_initial_tensors == 1:

    while tool_to_wp_connections <= max_tool_to_wp_connections:

        csv_dir = "/home/FORGING/input/graph_csv/world_" + str(tool_to_wp_connections) + "/"
        output_dir = "/home/FORGING/input/tensor_files/world_" + str(tool_to_wp_connections) + "/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        logging.debug('Generating initial tensors.')
        logging.debug('Total number of simulation steps: %s', len(episode_desc))

        full_one_hot_counter = 0
        full_tool_to_wp_mesh_counter = 0
        full_wp_to_tool_mesh_counter = 0

        # CYCLE THROUGH THE EPISODE DESCRIPTION DETAILS
        for i in range(len(episode_desc)):

            j = i+1

            sim_time = episode_desc.iloc[i, 0]
            try:
                next_sim_time = episode_desc.iloc[j, 0]
            except:
                next_sim_time = -99

            step_num = episode_desc.iloc[i, 1]
            logging.debug('Sim Time variable sim_time: %s', sim_time)

            if sim_time < 0:
                simulation_counter += 1
                timestep_counter = 0
                full_tool_to_wp_mesh_counter = 0
                full_wp_to_tool_mesh_counter = 0
                create_edges = 1
            else:
                timestep_counter += 1
                create_edges = 0

            # IDENTIFY GRAPH NUMBER TO ACQUIRE DETAILS
            if step_num < 10:
                graph_count = "00" + str(step_num)
            elif step_num < 100:
                graph_count = "0" + str(step_num)
            else:
                graph_count = str(step_num)

            node_filename = "all_vertices_" + graph_count + ".csv"
            logging.debug('Node Filename node_filename: %s', node_filename)

            node_file = csv_dir + node_filename
            node_details_df = pd.read_csv(node_file, usecols=['x', 'y', 'z', 'ID'])
            node_details_df['node'] = node_details_df.index

            workpiece_face_filename = "workpiece_surface_faces_" + graph_count + ".csv"
            workpiece_face_file = csv_dir + workpiece_face_filename
            tool_face_filename = "tool_surface_faces_" + graph_count + ".csv"
            tool_face_file = csv_dir + tool_face_filename

            tool_to_wp_world_face_filename = "tool_to_wp_world_faces_" + graph_count + ".csv"
            tool_to_wp_world_face_file = csv_dir + tool_to_wp_world_face_filename

            wp_to_tool_world_face_filename = "wp_to_tool_world_faces_" + graph_count + ".csv"
            wp_to_tool_world_face_file = csv_dir + wp_to_tool_world_face_filename

            if simulation_counter < 10:
                simulation_mesh_file_indicator = "00" + str(simulation_counter)
            elif simulation_counter < 100:
                simulation_mesh_file_indicator = "0" + str(simulation_counter)

            # Mesh Face Tensor contains the face information for the tool and workpiece
            mesh_face_tensor_filename = str(simulation_mesh_file_indicator) + "_mesh_face.pt"
            mesh_face_tensor_file = output_dir + mesh_face_tensor_filename

            # Workpiece Mesh Face Tensor contains the face information for the workpiece
            workpiece_mesh_face_tensor_filename = str(simulation_mesh_file_indicator) + "_workpiece_mesh_face.pt"
            workpiece_mesh_face_tensor_file = output_dir + workpiece_mesh_face_tensor_filename

            # Tool to WP World Face Tensor contains the face information between the tool and workpiece
            tool_to_wp_world_face_tensor_filename = str(simulation_mesh_file_indicator) + "_tool_to_wp_world_face.pt"
            tool_to_wp_world_face_tensor_file = output_dir + tool_to_wp_world_face_tensor_filename

            # WP to Tool World Face Tensor contains the face information between the tool and workpiece
            wp_to_tool_world_face_tensor_filename = str(simulation_mesh_file_indicator) + "_wp_to_tool_world_face.pt"
            wp_to_tool_world_face_tensor_file = output_dir + wp_to_tool_world_face_tensor_filename

            # Mesh Edge Tensor contains the edge information for the tool and workpiece
            mesh_edge_tensor_filename = str(simulation_mesh_file_indicator) + "_mesh_edge.pt"
            mesh_edge_tensor_file = output_dir + mesh_edge_tensor_filename

            # Workpiece Mesh Edge Tensor contains the edge information for the workpiece
            workpiece_mesh_edge_tensor_filename = str(simulation_mesh_file_indicator) + "_workpiece_mesh_edge.pt"
            workpiece_mesh_edge_tensor_file = output_dir + workpiece_mesh_edge_tensor_filename

            # Tool to WP World Edge Tensor contains the edge information between the tool and workpiece
            tool_to_wp_world_edge_tensor_filename = str(simulation_mesh_file_indicator) + "_tool_to_wp_world_edge.pt"
            tool_to_wp_world_edge_tensor_file = output_dir + tool_to_wp_world_edge_tensor_filename

            # WP to Tool World Edge Tensor contains the edge information between the workpiece and tool
            wp_to_tool_world_edge_tensor_filename = str(simulation_mesh_file_indicator) + "_wp_to_tool_world_edge.pt"
            wp_to_tool_world_edge_tensor_file = output_dir + wp_to_tool_world_edge_tensor_filename

            simulation_tool_to_wp_onehot_tensor_filename = str(simulation_mesh_file_indicator) + "_tool_to_wp_node_onehot.pt"
            simulation_tool_to_wp_onehot_tensor_file = output_dir + simulation_tool_to_wp_onehot_tensor_filename

            simulation_wp_to_tool_onehot_tensor_filename = str(simulation_mesh_file_indicator) + "_wp_to_tool_node_onehot.pt"
            simulation_wp_to_tool_onehot_tensor_file = output_dir + simulation_wp_to_tool_onehot_tensor_filename

            simulation_mesh_pos_tensor_filename = str(simulation_mesh_file_indicator) + "_node_pos.pt"
            simulation_mesh_pos_tensor_file = output_dir + simulation_mesh_pos_tensor_filename

            simulation_tool_to_wp_mesh_pos_tensor_filename = str(simulation_mesh_file_indicator) + "_tool_to_wp_node_pos.pt"
            simulation_tool_to_wp_mesh_pos_tensor_file = output_dir + simulation_tool_to_wp_mesh_pos_tensor_filename

            simulation_wp_to_tool_mesh_pos_tensor_filename = str(simulation_mesh_file_indicator) + "_wp_to_tool_node_pos.pt"
            simulation_wp_to_tool_mesh_pos_tensor_file = output_dir + simulation_wp_to_tool_mesh_pos_tensor_filename

            simulation_node_tool_to_wp_velocity_tensor_filename = str(simulation_mesh_file_indicator) + "_tool_to_wp_node_velo.pt"
            simulation_node_tool_to_wp_velocity_tensor_file = output_dir + simulation_node_tool_to_wp_velocity_tensor_filename

            simulation_node_wp_to_tool_velocity_tensor_filename = str(simulation_mesh_file_indicator) + "_wp_to_tool_node_velo.pt"
            simulation_node_wp_to_tool_velocity_tensor_file = output_dir + simulation_node_wp_to_tool_velocity_tensor_filename

            simulation_y_tool_to_wp_tensor_filename = str(simulation_mesh_file_indicator) + "_tool_to_wp_node_y.pt"
            simulation_y_tool_to_wp_tensor_file = output_dir + simulation_y_tool_to_wp_tensor_filename

            simulation_y_wp_to_tool_tensor_filename = str(simulation_mesh_file_indicator) + "_wp_to_tool_node_y.pt"
            simulation_y_wp_to_tool_tensor_file = output_dir + simulation_y_wp_to_tool_tensor_filename

            mesh_edge_attr_tensor_filename = str(simulation_mesh_file_indicator) + "_mesh_edge_attr.pt"
            simple_mesh_edge_attr_tensor_filename = str(simulation_mesh_file_indicator) + "_simp_mesh_edge_attr.pt"
            mesh_edge_attr_tensor_file = output_dir + mesh_edge_attr_tensor_filename
            simple_mesh_edge_attr_tensor_file = output_dir + simple_mesh_edge_attr_tensor_filename

            workpiece_mesh_edge_attr_tensor_filename = str(simulation_mesh_file_indicator) + "_workpiece_mesh_edge_attr.pt"
            simple_workpiece_mesh_edge_attr_tensor_filename = str(simulation_mesh_file_indicator) + "_simp_workpiece_mesh_edge_attr.pt"
            workpiece_mesh_edge_attr_tensor_file = output_dir + workpiece_mesh_edge_attr_tensor_filename
            simple_workpiece_mesh_edge_attr_tensor_file = output_dir + simple_workpiece_mesh_edge_attr_tensor_filename

            tool_to_wp_world_edge_attr_tensor_filename = str(simulation_mesh_file_indicator) + "_tool_to_wp_world_edge_attr.pt"
            simple_tool_to_wp_world_edge_attr_tensor_filename = str(simulation_mesh_file_indicator) + "_simp_tool_to_wp_world_edge_attr.pt"
            tool_to_wp_world_edge_attr_tensor_file = output_dir + tool_to_wp_world_edge_attr_tensor_filename
            simple_tool_to_wp_world_edge_attr_tensor_file = output_dir + simple_tool_to_wp_world_edge_attr_tensor_filename

            wp_to_tool_world_edge_attr_tensor_filename = str(simulation_mesh_file_indicator) + "_wp_to_tool_world_edge_attr.pt"
            simple_wp_to_tool_world_edge_attr_tensor_filename = str(simulation_mesh_file_indicator) + "_simp_wp_to_tool_world_edge_attr.pt"
            wp_to_tool_world_edge_attr_tensor_file = output_dir + wp_to_tool_world_edge_attr_tensor_filename
            simple_wp_to_tool_world_edge_attr_tensor_file = output_dir + simple_wp_to_tool_world_edge_attr_tensor_filename

            # IF THIS IS THE FIRST STEP IN A TIME SERIES,
            # BUILD A LIST OF ALL NODES IN THE EXTERNAL FACES

            if timestep_counter == 0:
                logging.debug('Building node list for: \n     %s, \n     %s, \n     %s, \n     %s', workpiece_face_file, tool_face_file, tool_to_wp_world_face_file, wp_to_tool_world_face_file)
                workpiece_face_details_df = pd.read_csv(workpiece_face_file, usecols=['node_1', 'node_2', 'node_3'])
                tool_face_details_df = pd.read_csv(tool_face_file, usecols=['node_1', 'node_2', 'node_3'])
                mesh_face_details_df = pd.concat([workpiece_face_details_df, tool_face_details_df])

                tool_to_wp_world_face_details_df = pd.read_csv(tool_to_wp_world_face_file, usecols=['node_1', 'node_2', 'node_3'])
                wp_to_tool_world_face_details_df = pd.read_csv(wp_to_tool_world_face_file, usecols=['node_1', 'node_2', 'node_3'])

                print("mesh_face_details_df SHAPE: ", mesh_face_details_df.shape)
                print("workpiece_face_details_df SHAPE: ", workpiece_face_details_df.shape)
                print("tool_to_wp_world_face_details_df SHAPE: ", tool_to_wp_world_face_details_df.shape)
                print("wp_to_tool_world_face_details_df SHAPE: ", wp_to_tool_world_face_details_df.shape)

                mesh_node_1_values = mesh_face_details_df["node_1"].to_list()
                mesh_node_2_values = mesh_face_details_df["node_2"].to_list()
                mesh_node_3_values = mesh_face_details_df["node_3"].to_list()

                workpiece_mesh_node_1_values = workpiece_face_details_df["node_1"].to_list()
                workpiece_mesh_node_2_values = workpiece_face_details_df["node_2"].to_list()
                workpiece_mesh_node_3_values = workpiece_face_details_df["node_3"].to_list()

                tool_to_wp_world_node_1_values = tool_to_wp_world_face_details_df["node_1"].to_list()
                tool_to_wp_world_node_2_values = tool_to_wp_world_face_details_df["node_2"].to_list()
                tool_to_wp_world_node_3_values = tool_to_wp_world_face_details_df["node_3"].to_list()

                wp_to_tool_world_node_1_values = wp_to_tool_world_face_details_df["node_1"].to_list()
                wp_to_tool_world_node_2_values = wp_to_tool_world_face_details_df["node_2"].to_list()
                wp_to_tool_world_node_3_values = wp_to_tool_world_face_details_df["node_3"].to_list()

                logging.debug('  Updating all_mesh_node_numbers and all_world_node_numbers.')
                all_mesh_node_numbers = sorted(set(mesh_node_1_values + mesh_node_2_values + mesh_node_3_values))
                all_workpiece_mesh_node_numbers = sorted(set(workpiece_mesh_node_1_values + workpiece_mesh_node_2_values + workpiece_mesh_node_3_values))
                all_tool_to_wp_world_node_numbers = sorted(set(tool_to_wp_world_node_1_values + tool_to_wp_world_node_2_values + tool_to_wp_world_node_3_values))
                all_tool_to_wp_node_numbers = sorted(set(all_mesh_node_numbers + all_tool_to_wp_world_node_numbers))
                all_wp_to_tool_world_node_numbers = sorted(set(wp_to_tool_world_node_1_values + wp_to_tool_world_node_2_values + wp_to_tool_world_node_3_values))
                all_wp_to_tool_node_numbers = sorted(set(all_mesh_node_numbers + all_wp_to_tool_world_node_numbers))

            logging.debug('  Number of mesh nodes in Simulation %s, Step %s: %s', simulation_counter, timestep_counter, len(all_mesh_node_numbers))
            logging.debug('  Number of workpiece mesh nodes in Simulation %s, Step %s: %s', simulation_counter, timestep_counter, len(all_workpiece_mesh_node_numbers))
            logging.debug('  Number of tool to wp world nodes in Simulation %s, Step %s: %s', simulation_counter, timestep_counter, len(all_tool_to_wp_world_node_numbers))
            logging.debug('  Number of wp to tool world nodes in Simulation %s, Step %s: %s', simulation_counter, timestep_counter, len(all_wp_to_tool_world_node_numbers))
            logging.debug('  Number of all wp to tool nodes in Simulation %s, Step %s: %s', simulation_counter, timestep_counter, len(all_wp_to_tool_node_numbers))
            logging.debug('  Number of all tool to wp nodes in Simulation %s, Step %s: %s', simulation_counter, timestep_counter, len(all_tool_to_wp_node_numbers))

            # BUILD ONE_HOT_NODE DETAILS
            if timestep_counter == 0:
                full_mesh_counter = 0

                # STORE EXPECTED ONE_HOT_COUNT INFORMATION FOR THIS SIMULATION
                one_hot_1_count = len(node_details_df.loc[node_details_df['ID'] == 1])
                one_hot_2_count = len(node_details_df.loc[node_details_df['ID'] == 2])
                expected_one_hot_1_count = one_hot_1_count
                expected_one_hot_2_count = one_hot_2_count

                for node_number in all_tool_to_wp_node_numbers:
                    node_info = node_details_df.loc[node_details_df['node'] == node_number]
                    node_id = node_info.iloc[0]['ID']

                    if node_id == 1:
                        node_one_hot_1 = 1
                        node_one_hot_2 = 0
                    elif node_id == 2:
                        node_one_hot_1 = 0
                        node_one_hot_2 = 1
                    one_hot_row = [node_one_hot_1, node_one_hot_2]

                    if node_number == all_tool_to_wp_node_numbers[0]:
                        tool_to_wp_one_hot_array = np.array(one_hot_row)
                    else:
                        tool_to_wp_one_hot_array = np.vstack([tool_to_wp_one_hot_array, one_hot_row])

                logging.debug('  Saving simulation tool to wp one-hot tensor to: %s', simulation_tool_to_wp_onehot_tensor_file)
                #logging.debug('    TOOL to WP ONE HOT ARRAY: %s', str(tool_to_wp_one_hot_array))
                logging.debug('    TOOL to WP ONE HOT ARRAY SHAPE: %s', str(tool_to_wp_one_hot_array.shape))

                for node_number in all_wp_to_tool_node_numbers:
                    node_info = node_details_df.loc[node_details_df['node'] == node_number]
                    node_id = node_info.iloc[0]['ID']

                    if node_id == 1:
                        node_one_hot_1 = 1
                        node_one_hot_2 = 0
                    elif node_id == 2:
                        node_one_hot_1 = 0
                        node_one_hot_2 = 1
                    one_hot_row = [node_one_hot_1, node_one_hot_2]

                    if node_number == all_wp_to_tool_node_numbers[0]:
                        wp_to_tool_one_hot_array = np.array(one_hot_row)
                    else:
                        wp_to_tool_one_hot_array = np.vstack([wp_to_tool_one_hot_array, one_hot_row])

                logging.debug('  Saving simulation wp to tool one-hot tensor to: %s', simulation_wp_to_tool_onehot_tensor_file)
                #logging.debug('    WP to TOOL ONE HOT ARRAY: %s', str(wp_to_tool_one_hot_array))
                logging.debug('    WP to TOOL ONE HOT ARRAY SHAPE: %s', str(wp_to_tool_one_hot_array.shape))

                torch.save(wp_to_tool_one_hot_array, simulation_wp_to_tool_onehot_tensor_file)

            # VALIDATE NODE COUNTS
            one_hot_1_count = len(node_details_df.loc[node_details_df['ID'] == 1])
            one_hot_2_count = len(node_details_df.loc[node_details_df['ID'] == 2])

            if expected_one_hot_1_count != one_hot_1_count:
                logging.debug('  Unable to proceed. Expected ID=1 count is not aligned.')
                exit(100)
            elif expected_one_hot_2_count != one_hot_2_count:
                logging.debug('  Unable to proceed. Expected ID=1 count is not aligned.')
                exit(100)
            else:
                logging.debug('  Node counts are aligned for each node ID. Proceeding.')

            ################################################################################################
            # BUILD THE MESH AND WORLD POSITION ARRAY (ALL IN ONE; ALL NODES IN MESH)
            node_index = 0

            for node_number in all_mesh_node_numbers:
                node_info = node_details_df.loc[node_details_df['node'] == node_number]
                node_id = node_info.iloc[0]['ID']

                node_x = node_info.iloc[0]['x']
                node_y = node_info.iloc[0]['y']
                node_z = node_info.iloc[0]['z']

                if node_id == 1:
                    node_one_hot_1 = 1
                    node_one_hot_2 = 0
                elif node_id == 2:
                    node_one_hot_1 = 0
                    node_one_hot_2 = 1

                mesh_pos_row = [node_x, node_y, node_z]
                node_row = [node_index, node_number, node_x, node_y, node_z]
                one_hot_row = [node_one_hot_1, node_one_hot_2]

                if node_index == 0:
                    mesh_pos_array = np.array(mesh_pos_row)
                    node_array = np.array(node_row)

                else:
                    mesh_pos_array = np.vstack([mesh_pos_array, mesh_pos_row])
                    node_array = np.vstack([node_array, node_row])

                node_index += 1

            logging.debug("Full Mesh Counter: %s", str(full_mesh_counter))
            if full_mesh_counter == 0:
                full_mesh_pos_array = np.array([mesh_pos_array])
                full_mesh_counter += 1
            else:
                full_mesh_pos_array = np.vstack([full_mesh_pos_array, np.array([mesh_pos_array])])

            logging.debug("FULL_MESH_POS_ARRAY SHAPE: %s", str(full_mesh_pos_array.shape))

            if display_graphs == 1:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(full_mesh_pos_array[0])
                o3d.visualization.draw_geometries([pcd], window_name="Full Mesh Point Cloud")


            if next_sim_time < 0:
                logging.debug('  Full Mesh Position Array Shape: %s', full_mesh_pos_array.shape)
                logging.debug('  Saving simulation mesh position tensor to: %s', simulation_mesh_pos_tensor_file)
                torch.save(full_mesh_pos_array, simulation_mesh_pos_tensor_file)

                full_mesh_elements = full_mesh_pos_array.shape[0]
                logging.debug("FULL MESH ELEMENTS: %s", str(full_mesh_elements))




            ################################################################################################
            # BUILD THE TOOL TO WP POSITION ARRAY
            element_counter = 0
            full_tool_to_wp_node_velocity_counter = 0
            full_wp_to_tool_node_velocity_counter = 0

            node_index = 0
            for node_number in all_tool_to_wp_node_numbers:
                node_info = node_details_df.loc[node_details_df['node'] == node_number]
                node_id = node_info.iloc[0]['ID']

                node_x = node_info.iloc[0]['x']
                node_y = node_info.iloc[0]['y']
                node_z = node_info.iloc[0]['z']

                if node_id == 1:
                    node_one_hot_1 = 1
                    node_one_hot_2 = 0
                elif node_id == 2:
                    node_one_hot_1 = 0
                    node_one_hot_2 = 1

                mesh_pos_row = [node_x, node_y, node_z]
                node_row = [node_index, node_number, node_x, node_y, node_z]
                one_hot_row = [node_one_hot_1, node_one_hot_2]

                if node_index == 0:
                    tool_to_wp_mesh_pos_array = np.array(mesh_pos_row)
                    node_array = np.array(node_row)

                else:
                    tool_to_wp_mesh_pos_array = np.vstack([tool_to_wp_mesh_pos_array, mesh_pos_row])
                    node_array = np.vstack([node_array, node_row])

                node_index += 1

            logging.debug("Full Tool to WP Mesh Counter: %s", str(full_tool_to_wp_mesh_counter))
            if full_tool_to_wp_mesh_counter == 0:
                full_tool_to_wp_mesh_pos_array = np.array([tool_to_wp_mesh_pos_array])
                full_tool_to_wp_mesh_counter += 1

            else:
                full_tool_to_wp_mesh_pos_array = np.vstack([full_tool_to_wp_mesh_pos_array, np.array([tool_to_wp_mesh_pos_array])])

            logging.debug("FULL_TOOL_TO_WP_MESH_POS_ARRAY SHAPE: %s", str(full_tool_to_wp_mesh_pos_array.shape))

            if display_graphs == 1:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(full_mesh_pos_array[0])
                o3d.visualization.draw_geometries([pcd], window_name="Workpiece Point Cloud")


            if next_sim_time < 0:
                logging.debug('  Full Tool to WP Mesh Position Array Shape: %s', full_tool_to_wp_mesh_pos_array.shape)
                logging.debug('  Saving simulation mesh position tensor to: %s', simulation_tool_to_wp_mesh_pos_tensor_file)
                torch.save(full_tool_to_wp_mesh_pos_array, simulation_tool_to_wp_mesh_pos_tensor_file)

                full_mesh_elements = full_tool_to_wp_mesh_pos_array.shape[0]
                logging.debug("FULL MESH ELEMENTS: %s", str(full_mesh_elements))

                element_counter = 0
                full_tool_to_wp_node_velocity_counter = 0
                full_wp_to_tool_node_velocity_counter = 0

                ################################################################################################
                # BUILD THE TOOL TO WP VELOCITY ARRAY
                while element_counter < (full_mesh_elements - 1):
                    current_mesh_element = full_tool_to_wp_mesh_pos_array[element_counter]
                    next_mesh_element = full_tool_to_wp_mesh_pos_array[element_counter + 1]

                    mesh_element_index = 0

                    for node_number in all_tool_to_wp_node_numbers:
                        node_info = node_details_df.loc[node_details_df['node'] == node_number]
                        node_id = node_info.iloc[0]['ID']

                        # CALCULATE THE NODE VELOCITY
                        current_x = current_mesh_element[mesh_element_index, 0]
                        next_x = next_mesh_element[mesh_element_index, 0]
                        current_y = current_mesh_element[mesh_element_index, 1]
                        next_y = next_mesh_element[mesh_element_index, 1]
                        current_z = current_mesh_element[mesh_element_index, 2]
                        next_z = next_mesh_element[mesh_element_index, 2]

                        node_velocity_x = next_x - current_x
                        node_velocity_y = next_y - current_y
                        node_velocity_z = next_z - current_z

                        # SET Y_ROW FOR ACTUAL NODE VELOCITY; USED FOR VALIDATION
                        y_row = [node_velocity_x, node_velocity_y, node_velocity_z]

                        # SET NODE VELOCITY FOR TOOL ONLY; LEAVE AT ZERO FOR WORKPIECE
                        if node_id == 2:
                            node_velocity_row = [node_velocity_x, node_velocity_y, node_velocity_z]
                        else:
                            node_velocity_row = [0.0, 0.0, 0.0]

                        # UPDATE NODE_VELOCITY_ARRAY AND Y_ARRAY WITH INFORMATION FROM VALUES ABOVE
                        if mesh_element_index == 0:
                            node_velocity_array = np.array(node_velocity_row)
                            y_array = np.array(y_row)
                        else:
                            node_velocity_array = np.vstack([node_velocity_array, node_velocity_row])
                            y_array = np.vstack([y_array, y_row])

                        mesh_element_index += 1

                    if full_tool_to_wp_node_velocity_counter == 0:
                        full_tool_to_wp_node_velocity_array = np.array([node_velocity_array])
                        full_y_array = np.array([y_array])
                        full_tool_to_wp_node_velocity_counter += 1
                    else:
                        full_tool_to_wp_node_velocity_array = np.vstack([full_tool_to_wp_node_velocity_array, np.array([node_velocity_array])])
                        full_y_array = np.vstack([full_y_array, np.array([y_array])])

                    element_counter += 1

                logging.debug('  Full Tool to WP Node Velocity Array Shape: %s', full_tool_to_wp_node_velocity_array.shape)
                logging.debug('  Saving simulation node tool to wp velocity tensor to: %s', simulation_node_tool_to_wp_velocity_tensor_file)

                torch.save(full_tool_to_wp_node_velocity_array, simulation_node_tool_to_wp_velocity_tensor_file)

                logging.debug('  Full Y Array Shape: %s', full_y_array.shape)
                logging.debug('  Saving simulation y tensor to: %s', simulation_y_tool_to_wp_tensor_file)
                torch.save(full_y_array, simulation_y_tool_to_wp_tensor_file)

            ################################################################################################
            # BUILD THE WP TO TOOL POSITION ARRAY

            node_index = 0
            for node_number in all_wp_to_tool_node_numbers:
                node_info = node_details_df.loc[node_details_df['node'] == node_number]
                node_id = node_info.iloc[0]['ID']

                node_x = node_info.iloc[0]['x']
                node_y = node_info.iloc[0]['y']
                node_z = node_info.iloc[0]['z']

                if node_id == 1:
                    node_one_hot_1 = 1
                    node_one_hot_2 = 0
                elif node_id == 2:
                    node_one_hot_1 = 0
                    node_one_hot_2 = 1

                mesh_pos_row = [node_x, node_y, node_z]
                node_row = [node_index, node_number, node_x, node_y, node_z]
                one_hot_row = [node_one_hot_1, node_one_hot_2]

                if node_index == 0:
                    wp_to_tool_mesh_pos_array = np.array(mesh_pos_row)
                    node_array = np.array(node_row)

                else:
                    wp_to_tool_mesh_pos_array = np.vstack([wp_to_tool_mesh_pos_array, mesh_pos_row])
                    node_array = np.vstack([node_array, node_row])

                node_index += 1

            logging.debug("Full WP to Tool Mesh Counter: %s", str(full_wp_to_tool_mesh_counter))
            if full_wp_to_tool_mesh_counter == 0:
                full_wp_to_tool_mesh_pos_array = np.array([wp_to_tool_mesh_pos_array])
                full_wp_to_tool_mesh_counter += 1

            else:
                full_wp_to_tool_mesh_pos_array = np.vstack([full_wp_to_tool_mesh_pos_array, np.array([wp_to_tool_mesh_pos_array])])

            #print("FULL_MESH_POS_ARRAY: ", full_mesh_pos_array)
            logging.debug("FULL_WP_TO_TOOL_MESH_POS_ARRAY SHAPE: %s", str(full_wp_to_tool_mesh_pos_array.shape))

            if display_graphs == 1:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(full_mesh_pos_array[0])
                o3d.visualization.draw_geometries([pcd], window_name="Workpiece Point Cloud")

            if next_sim_time < 0:
                logging.debug('  Full WP to Tool Mesh Position Array Shape: %s', full_wp_to_tool_mesh_pos_array.shape)
                logging.debug('  Saving simulation mesh position tensor to: %s', simulation_wp_to_tool_mesh_pos_tensor_file)
                torch.save(full_wp_to_tool_mesh_pos_array, simulation_wp_to_tool_mesh_pos_tensor_file)

                full_wp_to_tool_mesh_elements = full_wp_to_tool_mesh_pos_array.shape[0]

                element_counter = 0
                full_wp_to_tool_node_velocity_counter = 0

                ################################################################################################
                # BUILD THE WP TO TOOL VELOCITY ARRAY

                while element_counter < (full_wp_to_tool_mesh_elements - 1):
                    current_mesh_element = full_wp_to_tool_mesh_pos_array[element_counter]
                    next_mesh_element = full_wp_to_tool_mesh_pos_array[element_counter + 1]

                    mesh_element_index = 0

                    for node_number in all_wp_to_tool_node_numbers:
                        node_info = node_details_df.loc[node_details_df['node'] == node_number]
                        node_id = node_info.iloc[0]['ID']

                        # CALCULATE THE NODE VELOCITY
                        current_x = current_mesh_element[mesh_element_index, 0]
                        next_x = next_mesh_element[mesh_element_index, 0]
                        current_y = current_mesh_element[mesh_element_index, 1]
                        next_y = next_mesh_element[mesh_element_index, 1]
                        current_z = current_mesh_element[mesh_element_index, 2]
                        next_z = next_mesh_element[mesh_element_index, 2]

                        node_velocity_x = next_x - current_x
                        node_velocity_y = next_y - current_y
                        node_velocity_z = next_z - current_z

                        # SET Y_ROW FOR ACTUAL NODE VELOCITY; USED FOR VALIDATION
                        y_row = [node_velocity_x, node_velocity_y, node_velocity_z]

                        # SET NODE VELOCITY FOR TOOL ONLY; LEAVE AT ZERO FOR WORKPIECE
                        if node_id == 2:
                            node_velocity_row = [node_velocity_x, node_velocity_y, node_velocity_z]
                        else:
                            node_velocity_row = [0.0, 0.0, 0.0]

                        # UPDATE NODE_VELOCITY_ARRAY AND Y_ARRAY WITH INFORMATION FROM VALUES ABOVE
                        if mesh_element_index == 0:
                            node_velocity_array = np.array(node_velocity_row)
                            y_array = np.array(y_row)
                        else:
                            node_velocity_array = np.vstack([node_velocity_array, node_velocity_row])
                            y_array = np.vstack([y_array, y_row])

                        mesh_element_index += 1

                    if full_wp_to_tool_node_velocity_counter == 0:
                        full_wp_to_tool_node_velocity_array = np.array([node_velocity_array])
                        full_y_array = np.array([y_array])
                        full_wp_to_tool_node_velocity_counter += 1
                    else:
                        full_wp_to_tool_node_velocity_array = np.vstack([full_wp_to_tool_node_velocity_array, np.array([node_velocity_array])])
                        full_y_array = np.vstack([full_y_array, np.array([y_array])])

                    element_counter += 1

                logging.debug('  Full WP to Tool Node Velocity Array Shape: %s', full_wp_to_tool_node_velocity_array.shape)
                logging.debug('  Saving simulation node wp to tool velocity tensor to: %s', simulation_node_wp_to_tool_velocity_tensor_file)
                #print(full_node_velocity_array)

                torch.save(full_wp_to_tool_node_velocity_array, simulation_node_wp_to_tool_velocity_tensor_file)

                logging.debug('  Full Y Array Shape: %s', full_y_array.shape)
                logging.debug('  Saving simulation y tensor to: %s', simulation_y_wp_to_tool_tensor_file)
                torch.save(full_y_array, simulation_y_wp_to_tool_tensor_file)

            #############################################################
            # SAVE THE MESH EDGE INDEX DETAILS

            logging.debug('  Mesh Face Details Dataframe: \n%s', mesh_face_details_df)
            logging.debug('  Mesh Face Details Dataframe SHAPE: %s', mesh_face_details_df.shape)

            logging.debug('  tool_to_wp_world_face_details_df SHAPE: %s', tool_to_wp_world_face_details_df.shape)
            logging.debug('  wp_to_tool_world_face_details_df SHAPE: %s', wp_to_tool_world_face_details_df.shape)

            mesh_face_array = generate_face_array(mesh_face_details_df, node_array)
            logging.debug('  Mesh Face Array: \n%s', mesh_face_array)
            logging.debug('  Mesh Face Array Shape: %s', mesh_face_array.shape)

            workpiece_mesh_face_array = generate_face_array(workpiece_face_details_df, node_array)
            logging.debug('  Workpiece Mesh Face Array: \n%s', workpiece_mesh_face_array)
            logging.debug('  Workpiece Mesh Face Array Shape: %s', workpiece_mesh_face_array.shape)

            tool_to_wp_world_face_array = generate_face_array(tool_to_wp_world_face_details_df, node_array)
            #logging.debug('  Tool to WP World Face Array: \n%s', tool_to_wp_world_face_array)
            logging.debug('  Tool to WP World Face Array Shape: %s', tool_to_wp_world_face_array.shape)

            wp_to_tool_world_face_array = generate_face_array(wp_to_tool_world_face_details_df, node_array)
            #logging.debug('  WP to Tool World Face Array: \n%s', wp_to_tool_world_face_array)
            logging.debug('  WP to Tool World Face Array Shape: %s', wp_to_tool_world_face_array.shape)


            if create_edges == 1:

                #simulation_mesh_pos_array = np.array([mesh_pos_array])

                mesh_faces = tf.convert_to_tensor(mesh_face_array, dtype=tf.int32)
                workpiece_mesh_faces = tf.convert_to_tensor(workpiece_mesh_face_array, dtype=tf.int32)
                tool_to_wp_world_faces = tf.convert_to_tensor(tool_to_wp_world_face_array, dtype=tf.int32)
                wp_to_tool_world_faces = tf.convert_to_tensor(wp_to_tool_world_face_array, dtype=tf.int32)

                logging.debug('  Saving mesh face tensor to: %s', mesh_face_tensor_file)
                torch.save(mesh_faces, mesh_face_tensor_file)

                logging.debug('  Saving workpiece mesh face tensor to: %s', workpiece_mesh_face_tensor_file)
                torch.save(workpiece_mesh_faces, workpiece_mesh_face_tensor_file)

                logging.debug('  Saving tool to wp world face tensor to: %s', tool_to_wp_world_face_tensor_file)
                torch.save(tool_to_wp_world_faces, tool_to_wp_world_face_tensor_file)

                logging.debug('  Saving tool to wp world face tensor to: %s', wp_to_tool_world_face_tensor_file)
                torch.save(wp_to_tool_world_faces, wp_to_tool_world_face_tensor_file)

                mesh_edges = triangles_to_edges(mesh_faces)
                mesh_edge_index = torch.cat((torch.tensor(mesh_edges[0].numpy()).unsqueeze(0),
                                        torch.tensor(mesh_edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)

                workpiece_mesh_edges = triangles_to_edges(workpiece_mesh_faces)
                workpiece_mesh_edge_index = torch.cat((torch.tensor(workpiece_mesh_edges[0].numpy()).unsqueeze(0),
                                             torch.tensor(workpiece_mesh_edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)

                tool_to_wp_world_edges = triangles_to_edges(tool_to_wp_world_faces)
                tool_to_wp_world_edge_index = torch.cat((torch.tensor(tool_to_wp_world_edges[0].numpy()).unsqueeze(0),
                                        torch.tensor(tool_to_wp_world_edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)

                wp_to_tool_world_edges = triangles_to_edges(wp_to_tool_world_faces)
                wp_to_tool_world_edge_index = torch.cat((torch.tensor(wp_to_tool_world_edges[0].numpy()).unsqueeze(0),
                                        torch.tensor(wp_to_tool_world_edges[1].numpy()).unsqueeze(0)), dim=0).type(torch.long)

                #logging.debug("MESH EDGE INDEX: %s", str(mesh_edge_index))
                logging.debug("MESH EDGE INDEX SHAPE: %s", str(mesh_edge_index.shape))

                #logging.debug("WORKPIECE MESH EDGE INDEX: %s", str(workpiece_mesh_edge_index))
                logging.debug("WORKPIECE MESH EDGE INDEX SHAPE: %s", workpiece_mesh_edge_index.shape)

                #logging.debug("TOOL TO WP WORLD EDGE INDEX: %s", tool_to_wp_world_edge_index)
                logging.debug("TOOL TO WP WORLD EDGE INDEX SHAPE: %s", tool_to_wp_world_edge_index.shape)

                #logging.debug("WP TO TOOL WORLD EDGE INDEX: %s", wp_to_tool_world_edge_index)
                logging.debug("WP TO TOOL WORLD EDGE INDEX SHAPE: %s", wp_to_tool_world_edge_index.shape)

                logging.debug('  Saving mesh edge index tensor to: %s', mesh_edge_tensor_file)
                #print(mesh_edge_index)
                torch.save(mesh_edge_index, mesh_edge_tensor_file)

                logging.debug('  Saving workpiece mesh edge index tensor to: %s', workpiece_mesh_edge_tensor_file)
                #print(workpiece_mesh_edge_index)
                torch.save(workpiece_mesh_edge_index, workpiece_mesh_edge_tensor_file)

                logging.debug('  Saving tool to wp world edge index tensor to: %s', tool_to_wp_world_edge_tensor_file)
                #print(tool_to_wp_world_edge_index)
                torch.save(tool_to_wp_world_edge_index, tool_to_wp_world_edge_tensor_file)

                logging.debug('  Saving wp to tool world edge index tensor to: %s', wp_to_tool_world_edge_tensor_file)
                #print(wp_to_tool_world_edge_index)
                torch.save(wp_to_tool_world_edge_index, wp_to_tool_world_edge_tensor_file)
            #
            ##############################################################

            #############################################################
            # SAVE EDGE ATTRIBUTES

            mesh_edge_attributes = generate_edge_attributes(mesh_pos_array, mesh_edge_index, "mesh")
            workpiece_mesh_edge_attributes = generate_edge_attributes(mesh_pos_array, workpiece_mesh_edge_index, "mesh")
            tool_to_wp_world_edge_attributes = generate_edge_attributes(mesh_pos_array, tool_to_wp_world_edge_index, "world")
            wp_to_tool_world_edge_attributes = generate_edge_attributes(mesh_pos_array, wp_to_tool_world_edge_index, "world")

            simple_mesh_edge_attributes = simple_generate_edge_attributes(mesh_pos_array, mesh_edge_index)
            simple_workpiece_mesh_edge_attributes = simple_generate_edge_attributes(mesh_pos_array, workpiece_mesh_edge_index)
            simple_tool_to_wp_world_edge_attributes = simple_generate_edge_attributes(mesh_pos_array, tool_to_wp_world_edge_index)
            simple_wp_to_tool_world_edge_attributes = simple_generate_edge_attributes(mesh_pos_array, wp_to_tool_world_edge_index)

            print("MESH_EDGE_ATTR: ", mesh_edge_attributes)
            print("MESH_EDGE_ATTR SHAPE: ", mesh_edge_attributes.shape)
            print("WORKPIECE_MESH_EDGE_ATTR: ", workpiece_mesh_edge_attributes)
            print("WORKPIECE_MESH_EDGE_ATTR SHAPE: ", workpiece_mesh_edge_attributes.shape)
            print("TOOL_TO_WP_WORLD_EDGE_ATTR: ", tool_to_wp_world_edge_attributes)
            print("TOOL_TO_WP_WORLD_EDGE_ATTR SHAPE: ", tool_to_wp_world_edge_attributes.shape)
            print("WP_TO_TOOL_WORLD_EDGE_ATTR: ", wp_to_tool_world_edge_attributes)
            print("WP_TO_TOOL_WORLD_EDGE_ATTR SHAPE: ", wp_to_tool_world_edge_attributes.shape)

            if create_edges == 1:
                mesh_edge_attr_tensor = mesh_edge_attributes
                workpiece_mesh_edge_attr_tensor = workpiece_mesh_edge_attributes
                tool_to_wp_world_edge_attr_tensor = tool_to_wp_world_edge_attributes
                wp_to_tool_world_edge_attr_tensor = wp_to_tool_world_edge_attributes

                simple_mesh_edge_attr_tensor = simple_mesh_edge_attributes
                simple_workpiece_mesh_edge_attr_tensor = simple_workpiece_mesh_edge_attributes
                simple_tool_to_wp_world_edge_attr_tensor = simple_tool_to_wp_world_edge_attributes
                simple_wp_to_tool_world_edge_attr_tensor = simple_wp_to_tool_world_edge_attributes

            else:
                mesh_edge_attr_tensor = torch.cat((mesh_edge_attr_tensor, mesh_edge_attributes)).type(torch.float)
                workpiece_mesh_edge_attr_tensor = torch.cat((workpiece_mesh_edge_attr_tensor, workpiece_mesh_edge_attributes)).type(torch.float)
                tool_to_wp_world_edge_attr_tensor = torch.cat((tool_to_wp_world_edge_attr_tensor, tool_to_wp_world_edge_attributes)).type(torch.float)
                wp_to_tool_world_edge_attr_tensor = torch.cat((wp_to_tool_world_edge_attr_tensor, wp_to_tool_world_edge_attributes)).type(torch.float)

                simple_mesh_edge_attr_tensor = torch.cat((simple_mesh_edge_attr_tensor, simple_mesh_edge_attributes)).type(torch.float)
                simple_workpiece_mesh_edge_attr_tensor = torch.cat((simple_workpiece_mesh_edge_attr_tensor, simple_workpiece_mesh_edge_attributes)).type(torch.float)
                simple_tool_to_wp_world_edge_attr_tensor = torch.cat((simple_tool_to_wp_world_edge_attr_tensor, simple_tool_to_wp_world_edge_attributes)).type(torch.float)
                simple_wp_to_tool_world_edge_attr_tensor = torch.cat((simple_wp_to_tool_world_edge_attr_tensor, simple_wp_to_tool_world_edge_attributes)).type(torch.float)

            if next_sim_time < 0:
                logging.debug('  Saving mesh edge attribute index tensor to: %s', mesh_edge_attr_tensor_file)
                #print(mesh_edge_attr_tensor)
                logging.debug('Mesh Edge Attr Tensor Shape: %s', str(mesh_edge_attr_tensor.shape))
                torch.save(mesh_edge_attr_tensor, mesh_edge_attr_tensor_file)
                torch.save(simple_mesh_edge_attr_tensor, simple_mesh_edge_attr_tensor_file)

                logging.debug('  Saving workpiece mesh edge attribute index tensor to: %s', workpiece_mesh_edge_attr_tensor_file)
                #print(workpiece_mesh_edge_attr_tensor)
                torch.save(workpiece_mesh_edge_attr_tensor, workpiece_mesh_edge_attr_tensor_file)
                torch.save(simple_workpiece_mesh_edge_attr_tensor, simple_workpiece_mesh_edge_attr_tensor_file)

                logging.debug('  Saving tool to wp world edge attribute index tensor to: %s', tool_to_wp_world_edge_attr_tensor_file)
                #print(tool_to_wp_world_edge_attr_tensor)
                torch.save(tool_to_wp_world_edge_attr_tensor, tool_to_wp_world_edge_attr_tensor_file)
                torch.save(simple_tool_to_wp_world_edge_attr_tensor, simple_tool_to_wp_world_edge_attr_tensor_file)

                logging.debug('  Saving wp to tool world edge attribute index tensor to: %s', wp_to_tool_world_edge_attr_tensor_file)
                #print(wp_to_tool_world_edge_attr_tensor)
                torch.save(wp_to_tool_world_edge_attr_tensor, wp_to_tool_world_edge_attr_tensor_file)
                torch.save(simple_wp_to_tool_world_edge_attr_tensor, simple_wp_to_tool_world_edge_attr_tensor_file)

            #
            ##############################################################

        simulation_counter = -1
        tool_to_wp_connections += 1

exit(0)