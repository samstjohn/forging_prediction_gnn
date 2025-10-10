import torch

if torch.cuda.is_available():
    print("CUDA Version: ", torch.version.cuda)
    print("Pytorch Version: ", torch.__version__)

else:
    print("CUDA is not available.")

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch_scatter
import torch.nn as nn
import torch.optim as optim
from torch.nn import Linear, Sequential, LayerNorm, ReLU
from tqdm import trange

import os
import random
import numpy as np
import logging
import pandas as pd
import copy

import matplotlib.pyplot as plt


def normalize(to_normalize,mean_vec,std_vec):
    return (to_normalize-mean_vec)/std_vec

def unnormalize(to_unnormalize,mean_vec,std_vec):
    return to_unnormalize*std_vec+mean_vec

def get_stats(data_list):
    '''
    Method for normalizing processed datasets. Given the processed data_list,
    calculates the mean and standard deviation for the node features, edge features,
    and node outputs, and normalizes these using the calculated statistics.
    '''

    # Mean and std of the node features are calculated
    mean_vec_x=torch.zeros(data_list[0].x.shape[1:])
    std_vec_x=torch.zeros(data_list[0].x.shape[1:])

    #mean and std of the edge features are calculated
    mean_vec_edge=torch.zeros(data_list[0].edge_attr.shape[1:])
    std_vec_edge=torch.zeros(data_list[0].edge_attr.shape[1:])

    #mean and std of the output parameters are calculated
    mean_vec_y=torch.zeros(data_list[0].y.shape[1:])
    std_vec_y=torch.zeros(data_list[0].y.shape[1:])

    #Define the maximum number of accumulations to perform such that we do
    #not encounter memory issues
    max_accumulations = 10**6

    #Define a very small value for normalizing to
    eps=torch.tensor(1e-8)

    #Define counters used in normalization
    num_accs_x = 0
    num_accs_edge=0
    num_accs_y=0

    #Iterate through the data in the list to accumulate statistics
    for dp in data_list:

        #Add to the
        mean_vec_x+=torch.sum(dp.x,dim=0)
        std_vec_x+=torch.sum(dp.x**2,dim=0)
        num_accs_x+=dp.x.shape[0]

        mean_vec_edge+=torch.sum(dp.edge_attr,dim=0)
        std_vec_edge+=torch.sum(dp.edge_attr**2,dim=0)
        num_accs_edge+=dp.edge_attr.shape[0]

        mean_vec_y+=torch.sum(dp.y,dim=0)
        std_vec_y+=torch.sum(dp.y**2,dim=0)
        num_accs_y+=dp.y.shape[0]

        if(num_accs_x>max_accumulations or num_accs_edge>max_accumulations): # or num_accs_y>max_accumulations):
            break

    mean_vec_x = mean_vec_x/num_accs_x
    std_vec_x = torch.maximum(torch.sqrt(std_vec_x/num_accs_x - mean_vec_x**2),eps)

    mean_vec_edge = mean_vec_edge/num_accs_edge
    std_vec_edge = torch.maximum(torch.sqrt(std_vec_edge/num_accs_edge - mean_vec_edge**2),eps)

    mean_vec_y = mean_vec_y/num_accs_y
    std_vec_y = torch.maximum(torch.sqrt(std_vec_y/num_accs_y - mean_vec_y**2),eps)

    mean_std_list=[mean_vec_x,std_vec_x,mean_vec_edge,std_vec_edge,mean_vec_y,std_vec_y]

    return mean_std_list

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


#def train(dataset, device, stats_list, args):
def train(train_dataset, test_dataset, device, stats_list, loo, args):
    '''
    Performs a training loop on the dataset for MeshGraphNets. Also calls
    test and validation functions.
    '''

    df = pd.DataFrame(columns=['epoch', 'train_loss', 'test_loss'])

    # Define the model name for saving
    model_name = 'MODEL_LOO_SIM_' + str(loo) + '_NL_' + str(args.num_layers) + '_BS_' + str(args.batch_size) + \
                 '_HD_' + str(args.hidden_dim) + '_EP_' + str(args.epochs) + '_WD_' + str(args.weight_decay) + \
                 '_LR_' + str(args.lr) + '_SHUFF_' + str(args.shuffle) + '_TR_' + str(args.train_size) + '_TE_' + str(args.test_size)

    # torch_geometric DataLoaders are used for handling the data of lists of graphs
    #loader = DataLoader(dataset[:args.train_size], batch_size=args.batch_size, shuffle=False)
    #test_loader = DataLoader(dataset[args.train_size:], batch_size=args.batch_size, shuffle=False)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # The statistics of the data are decomposed
    [mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y] = stats_list
    (mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y) = (mean_vec_x.to(device),
                                                                                   std_vec_x.to(device),
                                                                                   mean_vec_edge.to(device),
                                                                                   std_vec_edge.to(device),
                                                                                   mean_vec_y.to(device),
                                                                                   std_vec_y.to(device))

    # build model
    num_node_features = train_dataset[0].x.shape[1]
    num_edge_features = train_dataset[0].edge_attr.shape[1]
    num_classes = 3  # the dynamic variables have the shape of 3 (velocity)

    model = MeshGraphNet(num_node_features, num_edge_features, args.hidden_dim, num_classes,
                         args).to(device)
    scheduler, opt = build_optimizer(args, model.parameters())

    # train
    losses = []
    test_losses = []
    best_test_loss = np.inf
    best_model = None
    for epoch in trange(args.epochs, desc="Training", unit="Epochs"):
        total_loss = 0
        model.train()
        num_loops = 0
        for batch in loader:
            # Note that normalization must be done before it's called. The unnormalized
            # data needs to be preserved in order to correctly calculate the loss
            batch = batch.to(device)
            opt.zero_grad()  # zero gradients each time
            pred = model(batch, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss = model.loss(pred, batch, mean_vec_y, std_vec_y)

            loss.backward()  # backpropagate loss
            opt.step()
            total_loss += loss.item()

            num_loops += 1
        total_loss /= num_loops
        losses.append(total_loss)

        # Every fifth epoch, calculate test loss
        if epoch % 5 == 0:

            test_loss, _ = test(test_loader, device, model, mean_vec_x, std_vec_x, mean_vec_edge,
                                    std_vec_edge, mean_vec_y, std_vec_y, args.save_velo_val)

            test_losses.append(test_loss.item())

            # saving model
            if not os.path.isdir(args.checkpoint_dir):
                os.mkdir(args.checkpoint_dir)

            #CSV_PATH = os.path.join(args.checkpoint_dir, model_name + '.csv')
            #df.to_csv(CSV_PATH, index=False)

            # save the model if the current one is better than the previous best
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model = copy.deepcopy(model)

        else:
            test_losses.append(test_losses[-1])


        df_newline_dict = {'epoch': epoch, 'train_loss': losses[-1],
                            'test_loss': test_losses[-1]}
        df_newline = pd.DataFrame(df_newline_dict, index=[0])
        df = pd.concat([df, df_newline], ignore_index=True)

        if epoch % 5 == 0:
            print("  train loss", str(round(total_loss, 5)), " | test loss", str(round(test_loss.item(), 5)))

            if (args.save_best_model):
                PATH = os.path.join(args.checkpoint_dir, model_name + '.pt')
                torch.save(best_model.state_dict(), PATH)

    CSV_PATH = os.path.join(args.checkpoint_dir, model_name + '.csv')
    df.to_csv(CSV_PATH, index=False)

    return test_losses, losses, best_model, best_test_loss, test_loader


def test(loader, device, test_model,
         mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge, mean_vec_y, std_vec_y, is_validation,
         delta_t=0.01, save_model_preds=False, model_type=None):
    '''
    Calculates test set losses and validation set errors.
    '''

    loss = 0
    velo_rmse = 0
    num_loops = 0

    for data in loader:
        data = data.to(device)
        with torch.no_grad():

            # calculate the loss for the model given the test set
            pred = test_model(data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge)
            loss += test_model.loss(pred, data, mean_vec_y, std_vec_y)

            # calculate validation error if asked to
            if (is_validation):
                # Like for the MeshGraphNets model, calculate the mask over which we calculate
                # flow loss and add this calculated RMSE value to our val error
                toolpiece = torch.tensor(2)
                workpiece = torch.tensor(1)

                loss_mask = torch.argmax(data.x[:, 2:], dim=1) == torch.tensor(1)

                eval_velo = data.x[:, 0:3] + unnormalize(pred[:], mean_vec_y, std_vec_y) * delta_t
                gs_velo = data.x[:, 0:3] + data.y[:] * delta_t

                error = torch.sum((eval_velo - gs_velo) ** 2, axis=1)

                velo_rmse += torch.sqrt(torch.mean(error[loss_mask]))

        num_loops += 1
        # if velocity is evaluated, return velo_rmse as 0
    return loss / num_loops, velo_rmse / num_loops


def save_plots(args, loss_image_title, losses, test_losses):

    file_prepend = 'SIM_' + loss_image_title.split(' ')[-1] + '_'
    no_title_prepend = file_prepend + 'NO_TITLE_'

    plot_file_name = file_prepend + '_NL_' + str(args.num_layers) + '_BS_' + \
        str(args.batch_size) + '_HD_' + str(args.hidden_dim) + '_EP_' + str(args.epochs) + '_WD_' + \
        str(args.weight_decay) + '_LR_' + str(args.lr) + '_SHUFF_' + str(args.shuffle) + '_TR_' + \
        str(args.train_size) + '_TE_' + str(args.test_size)

    no_title_plot_file_name = no_title_prepend + '_NL_' + str(args.num_layers) + '_BS_' + \
        str(args.batch_size) + '_HD_' + str(args.hidden_dim) + '_EP_' + str(args.epochs) + '_WD_' + \
        str(args.weight_decay) + '_LR_' + str(args.lr) + '_SHUFF_' + str(args.shuffle) + '_TR_' + \
        str(args.train_size) + '_TE_' + str(args.test_size)

    if not os.path.isdir(args.postprocess_dir):
        os.mkdir(args.postprocess_dir)

    PLOT_FILE_PATH = os.path.join(args.postprocess_dir, plot_file_name + '.eps')
    NO_TITLE_PLOT_FILE_PATH = os.path.join(args.postprocess_dir, no_title_plot_file_name + '.eps')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.sans-serif'] = "Times New Roman"

    f1 = plt.figure(figsize=(6,6))
    plt.title(loss_image_title)
    plt.plot(losses, label="Training Loss", color='#156082')
    plt.plot(test_losses, label="Test Loss", color='#FFC000')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(top=5)
    plt.legend()
    plt.show()
    f1.savefig(PLOT_FILE_PATH, bbox_inches='tight', format='eps')

    f2 = plt.figure(figsize=(6,6))
    plt.title('')
    plt.plot(losses, label="Training Loss", color='#156082')
    plt.plot(test_losses, label="Test Loss", color='#FFC000')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(top=5)
    plt.legend()
    plt.show()
    f2.savefig(NO_TITLE_PLOT_FILE_PATH, bbox_inches='tight', format='eps')

#######################################################################################################
# CLASSES DEFINED
#######################################################################################################

class MeshGraphNet(torch.nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, output_dim, args, emb=False):
        super(MeshGraphNet, self).__init__()
        """
        MeshGraphNet model. This model is built upon Deepmind's 2021 paper.
        This model consists of three parts: (1) Preprocessing: encoder (2) Processor
        (3) postproccessing: decoder. Encoder has an edge and node decoders respectively.
        Processor has two processors for edge and node respectively. Note that edge attributes have to be
        updated first. Decoder is only for nodes.

        Input_dim: dynamic variables + node_type + node_position
        Hidden_dim: 128 in deepmind's paper
        Output_dim: dynamic variables: velocity changes (1)

        """

        self.num_layers = args.num_layers

        # encoder convert raw inputs into latent embeddings
        self.node_encoder = Sequential(Linear(input_dim_node, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       LayerNorm(hidden_dim))

        self.edge_encoder = Sequential(Linear(input_dim_edge, hidden_dim),
                                       ReLU(),
                                       Linear(hidden_dim, hidden_dim),
                                       LayerNorm(hidden_dim)
                                       )

        self.processor = nn.ModuleList()
        assert (self.num_layers >= 1), 'Number of message passing layers is not >=1'

        processor_layer = self.build_processor_model()
        for _ in range(self.num_layers):
            self.processor.append(processor_layer(hidden_dim, hidden_dim))

        # decoder: only for node embeddings
        self.decoder = Sequential(Linear(hidden_dim, hidden_dim),
                                  ReLU(),
                                  Linear(hidden_dim, output_dim)
                                  )

    def build_processor_model(self):
        return ProcessorLayer

    def forward(self, data, mean_vec_x, std_vec_x, mean_vec_edge, std_vec_edge):
        """
        Encoder encodes graph (node/edge features) into latent vectors (node/edge embeddings)
        The return of processor is fed into the processor for generating new feature vectors
        """
        #x, edge_index, edge_attr, pressure = data.x, data.edge_index, data.edge_attr, data.p
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = normalize(x, mean_vec_x, std_vec_x)
        edge_attr = normalize(edge_attr, mean_vec_edge, std_vec_edge)

        # Step 1: encode node/edge features into latent node/edge embeddings
        x = self.node_encoder(x)  # output shape is the specified hidden dimension

        edge_attr = self.edge_encoder(edge_attr)  # output shape is the specified hidden dimension

        # step 2: perform message passing with latent node/edge embeddings
        for i in range(self.num_layers):
            x, edge_attr = self.processor[i](x, edge_index, edge_attr)

        # step 3: decode latent node embeddings into physical quantities of interest

        return self.decoder(x)

    def loss(self, pred, inputs, mean_vec_y, std_vec_y):
        # Define the node types that we calculate loss for
        toolpiece = torch.tensor(2)
        workpiece = torch.tensor(1)

        # Get the loss mask for the nodes of the types we calculate loss for
        loss_mask = torch.argmax(inputs.x[:, 2:], dim=1) == torch.tensor(1)

        # Normalize labels with dataset statistics
        labels = normalize(inputs.y, mean_vec_y, std_vec_y)

        # Find sum of square errors
        error = torch.sum((labels - pred) ** 2, axis=1)

        # Root and mean the errors for the nodes we calculate loss for
        loss = torch.sqrt(torch.mean(error[loss_mask]))

        return loss


class ProcessorLayer(MessagePassing):
    def __init__(self, in_channels, out_channels,  **kwargs):
        super(ProcessorLayer, self).__init__(  **kwargs )
        """
        in_channels: dim of node embeddings [128], out_channels: dim of edge embeddings [128]

        """

        # Note that the node and edge encoders both have the same hidden dimension
        # size. This means that the input of the edge processor will always be
        # three times the specified hidden dimension
        # (input: adjacent node embeddings and self embeddings)
        self.edge_mlp = Sequential(Linear( 3* in_channels , out_channels),
                                   ReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))

        self.node_mlp = Sequential(Linear( 2* in_channels , out_channels),
                                   ReLU(),
                                   Linear( out_channels, out_channels),
                                   LayerNorm(out_channels))


        self.reset_parameters()

    def reset_parameters(self):
        """
        reset parameters for stacked MLP layers
        """
        self.edge_mlp[0].reset_parameters()
        self.edge_mlp[2].reset_parameters()

        self.node_mlp[0].reset_parameters()
        self.node_mlp[2].reset_parameters()

    def forward(self, x, edge_index, edge_attr, size = None):
        """
        Handle the pre and post-processing of node features/embeddings,
        as well as initiates message passing by calling the propagate function.

        Note that message passing and aggregation are handled by the propagate
        function, and the update

        x has shpae [node_num , in_channels] (node embeddings)
        edge_index: [2, edge_num]
        edge_attr: [E, in_channels]

        """

        out, updated_edges = self.propagate(edge_index, x = x, edge_attr = edge_attr, size = size) # out has the shape of [E, out_channels]

        updated_nodes = torch.cat([x,out],dim=1)        # Complete the aggregation through self-aggregation

        updated_nodes = x + self.node_mlp(updated_nodes) # residual connection

        return updated_nodes, updated_edges

    def message(self, x_i, x_j, edge_attr):
        """
        source_node: x_i has the shape of [E, in_channels]
        target_node: x_j has the shape of [E, in_channels]
        target_edge: edge_attr has the shape of [E, out_channels]

        The messages that are passed are the raw embeddings. These are not processed.
        """

        updated_edges=torch.cat([x_i, x_j, edge_attr], dim = 1) # tmp_emb has the shape of [E, 3 * in_channels]
        updated_edges=self.edge_mlp(updated_edges)+edge_attr

        return updated_edges

    def aggregate(self, updated_edges, edge_index, dim_size = None):
        """
        First we aggregate from neighbors (i.e., adjacent nodes) through concatenation,
        then we aggregate self message (from the edge itself). This is streamlined
        into one operation here.
        """

        # The axis along which to index number of nodes.
        node_dim = 0

        out = torch_scatter.scatter(updated_edges, edge_index[0, :], dim=node_dim, reduce = 'sum')

        return out, updated_edges

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

#######################################################################################################
# MAIN PROGRAM BEGINS HERE
#######################################################################################################

logging.basicConfig(level = logging.INFO, format = '%(asctime)s:%(levelname)s: %(message)s')


for args in [
        {'model_type': 'meshgraphnet',
         'num_layers': 10,
         'batch_size': 16,
         'hidden_dim': 10,
         #'epochs': 5000,
         #'epochs': 1000,
         'epochs': 750,
         'opt': 'adam',
         'opt_scheduler': 'none',
         'opt_restart': 0,
         'weight_decay': 5e-4,
         'lr': 0.001,
         'train_size': 45,
         'test_size': 10,
         'device':'cuda',
         'shuffle': False,
         'save_velo_val': False,
         'save_best_model': True,
         'checkpoint_dir': './best_models/',
         'postprocess_dir': './3d_loss_plots/'},
    ]:
        args = objectview(args)

#To ensure reproducibility the best we can, here we control the sources of
#randomness by seeding the various random number generators used in this Colab
#For more information, see: https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(5)  #Torch
random.seed(5)        #Python
np.random.seed(5)     #NumPy



#dataset = torch.load(file_path)[:(args.train_size+args.test_size)]
tool_to_wp_connections = 2

root_dir = "/home/FORGING/input/"
episode_file = root_dir + "graph_csv/episode_summary.csv"
episode_desc = pd.read_csv(episode_file)
tensor_dir = "/home/FORGING/input/tensor_files/" + str(tool_to_wp_connections) + "/"

logging.info('Episode Description \n--------------------------------------\n %s\n', episode_desc)

# Get the number of simulations executed
simulation_counter = 0
for i in range(len(episode_desc)):
    sim_time = episode_desc.iloc[i, 0]
    if sim_time < 0:
        simulation_counter += 1


for i in range(simulation_counter):


    if i < 10:
        simulation_mesh_file_indicator = "00" + str(i)
    elif i < 100:
        simulation_mesh_file_indicator = "0" + str(i)

    globals()[f"data_list_{simulation_mesh_file_indicator}"] = []

    logging.info('Creating data list for simulation %s', simulation_mesh_file_indicator)

    simulation_onehot_tensor_filename = str(simulation_mesh_file_indicator) + "_sim_onehot.pt"
    simulation_onehot_tensor_file = tensor_dir + simulation_onehot_tensor_filename

    simulation_node_velocity_tensor_filename = str(simulation_mesh_file_indicator) + "_sim_node_velo.pt"
    simulation_node_velocity_tensor_file = tensor_dir + simulation_node_velocity_tensor_filename

    simulation_y_tensor_filename = str(simulation_mesh_file_indicator) + "_sim_y.pt"
    simulation_y_tensor_file = tensor_dir + simulation_y_tensor_filename

    simulation_face_tensor_filename = str(simulation_mesh_file_indicator) + "_sim_face.pt"
    simulation_face_tensor_file = tensor_dir + simulation_face_tensor_filename

    simulation_edge_tensor_filename = str(simulation_mesh_file_indicator) + "_sim_edge.pt"
    simulation_edge_tensor_file = tensor_dir + simulation_edge_tensor_filename

    simulation_edge_attr_tensor_filename = str(simulation_mesh_file_indicator) + "_sim_edge_attr.pt"
    simulation_edge_attr_tensor_file = tensor_dir + simulation_edge_attr_tensor_filename

    simulation_onehot_array = torch.load(simulation_onehot_tensor_file)
    simulation_node_velocity_array = torch.load(simulation_node_velocity_tensor_file)
    simulation_y_array = torch.load(simulation_y_tensor_file)

    simulation_onehot_tensor = torch.from_numpy(simulation_onehot_array)
    simulation_node_velocity_tensor = torch.from_numpy(simulation_node_velocity_array)
    simulation_y_tensor = torch.from_numpy(simulation_y_array)

    #simulation_face_tensor = torch.from_numpy(simulation_face_tensor_file)
    simulation_face_tensor = torch.load(simulation_face_tensor_file)


    simulation_edge_index_tensor = torch.load(simulation_edge_tensor_file)
    simulation_edge_attr_tensor = torch.load(simulation_edge_attr_tensor_file)

    # Find the number of timesteps in this tensor
    simulation_timesteps = simulation_node_velocity_tensor.shape[0]

    data_list = []

    for j in range(simulation_timesteps):

        simulation_node_timestep_velocity_tensor = simulation_node_velocity_tensor[j]
        edge_attr_tensor = simulation_edge_attr_tensor[j]

        y_tensor = simulation_y_tensor[j]

        timestep_x = torch.cat((simulation_node_timestep_velocity_tensor, simulation_onehot_tensor), dim=-1).type(torch.float)
        timestep_x_array = np.array(timestep_x)

        simulation_x_array = []
        simulation_x_array = np.array(timestep_x_array)

        simulation_x_tensor = torch.from_numpy(simulation_x_array)

        # x is the combination of the 3D velocity and one-hot tensor [velo_x, velo_y, velo_z, id=1, id=2] x number of nodes
        # edge_index is the 2D representation of edges in the mesh [ [1st node of edge x number of nodes] [2nd node of edge x number of nodes] ]
        # edge_attr is the edge distances [ u_ij_x, u_ij_y, u_ij_z, |norm(u_ij)| ] x number of edges
        # y is the actual velocity for all nodes
        data_list.append(Data(x=simulation_x_tensor, edge_index=simulation_edge_index_tensor,
                              edge_attr=edge_attr_tensor, y=y_tensor))

    globals()[f"data_list_{simulation_mesh_file_indicator}"] = data_list

logging.info('Data list creation complete.')

for k, v in list(globals().items()):
    if k.startswith("data_list_"):
        rowcount = len(v)
        if rowcount < 10:
            rowcount = "  " + str(rowcount)
        elif rowcount < 100:
            rowcount = " " + str(rowcount)

        logging.info('  %s: %s entries with dimensions %s', k, rowcount, v[1])




for i in range(simulation_counter):

    overall_data_list = []
    train_data_list = []
    test_data_list = []

    if i < 10:
        simulation_mesh_file_indicator = "00" + str(i)
    elif i < 100:
        simulation_mesh_file_indicator = "0" + str(i)

    logging.info('Starting LOO Training for %s', simulation_mesh_file_indicator)
    loo = simulation_mesh_file_indicator

    for k, v in list(globals().items()):
        if k.startswith("data_list_"):
            if k.startswith(f"data_list_{simulation_mesh_file_indicator}"):
                logging.info('  LOO test data list:  %s', k)
                sim_number = k.split("_")[-1].lstrip('0')
                if sim_number == '':
                    sim_number = 0
                logging.info('  LOO simulation number: %s', sim_number)
                loss_image_title = "GNN LOOCV Loss: Simulation " + str(sim_number)
                test_data_list = v
            else:
                if len(train_data_list) == 0:
                    train_data_list = v
                else:
                    train_data_list = train_data_list + v

            if len(overall_data_list) == 0:
                overall_data_list = v
            else:
                overall_data_list = overall_data_list + v

    logging.info('  Overall data list length %s', len(overall_data_list))
    logging.info('  Train data list length %s', len(train_data_list))
    logging.info('  Test data list length %s', len(test_data_list))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    logging.info('  Proceeding with %s device.', device)

    stats_list = get_stats(overall_data_list)

    #test_losses, losses, velo_val_losses, best_model, best_test_loss, test_loader = train(train_data_list, test_data_list, device, stats_list, loo, args)
    test_losses, losses, best_model, best_test_loss, test_loader = train(train_data_list, test_data_list, device, stats_list, loo, args)


    save_plots(args, loss_image_title, losses, test_losses)

exit(0)