notebook_mode = False
viz_mode = False
TESTING=False

import os
import json
import argparse
import time

import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

import sys
from sklearn.utils.class_weight import compute_class_weight

from tensorboardX import SummaryWriter
from fastprogress import master_bar, progress_bar

# Remove warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

from config import *
from utils.graph_utils import *
from utils.google_tsp_reader import GoogleTSPReader
from utils.plot_utils import *
from models.gcn_model import ResidualGatedGCNModel
from utils.model_utils import *

import copy

from datetime import datetime


if notebook_mode==False:
    parser = argparse.ArgumentParser(description='gcn_tsp_parser')
    parser.add_argument('-c','--config', type=str, default="configs/gsize/meta_config.json")
    parser.add_argument('--Testing', type=str, default="False")
    args = parser.parse_args()
    config_path_meta = args.config
    TESTING = True if args.Testing == "True" else False
    print(" TESTING MODE :  ", TESTING)


config_meta = get_config(config_path_meta)
print("Loaded {}:\n{}".format(config_path_meta, config_meta))

tasks_list = [10,20,30,50]
tasks_lists_test = [100]

dict_tasks_configs = {}
dict_test_tasks_config = {}

for task in tasks_list:
    config_path_tasks = "configs/size/"+'tsp'+str(task)+'.json'
    config_task = get_config(config_path_tasks)
    config_task.gpu_id=0
    config_task.beam_size = 1
    dict_tasks_configs[str(task)] = config_task
    print("Loaded {}:\n{}".format(config_path_tasks, config_task))


for task in tasks_lists_test:
    config_path_tasks = "configs/size/"+'tsp'+str(task)+'.json'
    config_task = get_config(config_path_tasks)
    config_task.gpu_id=0
    config_task.beam_size = 1
    dict_test_tasks_config[str(task)] = config_task
    print("Loaded {}:\n{}".format(config_path_tasks, config_task))


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

if torch.cuda.is_available():
  print("CUDA available")
  dtypeFloat = torch.cuda.FloatTensor
  dtypeLong = torch.cuda.LongTensor
  torch.cuda.manual_seed(1)
else:
    print("CUDA not available")
    dtypeFloat = torch.FloatTensor
    dtypeLong = torch.LongTensor
    torch.manual_seed(1)



def train_one_step(net_meta,net_task, config_task, config_meta, master_bar):
    # Set training mode

    net = net_task
    net.train()
    optimizer_task = torch.optim.Adam(net.parameters(), lr=config_task.learning_rate)

    presever_old_weights_meta = net_meta.state_dict()

    # Assign parameters
    num_nodes = config_task.num_nodes
    num_neighbors = config_task.num_neighbors
    batch_size = config_meta.batch_size
    batches_per_epoch = config_meta.batches_per_epoch # number of fine_tuning steps
    accumulation_steps = config_task.accumulation_steps
    train_filepath = config_task.train_filepath

    # Load TSP data
    dataset_support = GoogleTSPReader(num_nodes, num_neighbors, batch_size, train_filepath, split_details={'split_percentage':50,'split_id':0})
    print("dataset support size" , len(dataset_support))
    if batches_per_epoch != -1:
        batches_per_epoch = min(batches_per_epoch, dataset_support.max_iter)
    else:
        batches_per_epoch = dataset_support.max_iter

    print("batches per epoch", batches_per_epoch)
    # Convert dataset to iterable
    dataset_support = iter(dataset_support)
    
    # Initially set loss class weights as None
    edge_cw = None

    # Initialize running data
    running_loss = 0.0
    running_pred_tour_len = 0.0
    running_gt_tour_len = 0.0
    running_nb_data = 0
    running_nb_batch = 0

    start_epoch = time.time()
    for batch_num in progress_bar(range(batches_per_epoch), parent=master_bar):
        # Generate a batch of TSPs
        try:
            batch = next(dataset_support)
        except StopIteration:
            break

        # Convert batch to torch Variables
        x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
        x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
        x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
        x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
        y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
        y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)
        
        # Compute class weights (if uncomputed)
        if type(edge_cw) != torch.Tensor:
            edge_labels = y_edges.cpu().numpy().flatten()
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)
        
        # Forward pass
        y_preds, loss = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
        loss = loss.mean()  # Take mean of loss across multiple GPUs
        loss = loss / accumulation_steps  # Scale loss by accumulation steps
        loss.backward()

        # Backward pass
        if (batch_num+1) % accumulation_steps == 0:
            optimizer_task.step()
            optimizer_task.zero_grad()

        # Compute error metrics and mean tour lengths
        # err_edges, err_tour, err_tsp, tour_err_idx, tsp_err_idx = edge_error(y_preds, y_edges, x_edges)
        pred_tour_len = mean_tour_len_edges(x_edges_values, y_preds)
        gt_tour_len = np.mean(batch.tour_len)

        # Update running data
        running_nb_data += batch_size
        running_loss += batch_size* loss.data.item()* accumulation_steps  # Re-scale loss
        running_pred_tour_len += batch_size* pred_tour_len
        running_gt_tour_len += batch_size* gt_tour_len
        running_nb_batch += 1

        # Log intermediate statistics
        result = ('loss:{loss:.4f} pred_tour_len:{pred_tour_len:.3f} gt_tour_len:{gt_tour_len:.3f}'.format(
            loss=running_loss/running_nb_data,
            pred_tour_len=running_pred_tour_len/running_nb_data,
            gt_tour_len=running_gt_tour_len/running_nb_data))
        master_bar.child.comment = result


    print("meta part ")
    net_meta.load_state_dict(net.state_dict())

    # Initially set loss class weights as None
    edge_cw = None

    # Initialize running data
    running_loss = 0.0
    running_pred_tour_len = 0.0
    running_gt_tour_len = 0.0
    running_nb_data = 0
    running_nb_batch = 0

    num_query_batches = 1

    dataset_query = GoogleTSPReader(num_nodes, num_neighbors, batch_size, train_filepath, split_details={'split_percentage':50,'split_id':1})

    print("SIZE of query dataset ", len(dataset_query))
    # Convert dataset to iterable
    dataset_query = iter(dataset_query)

    print("num query batches ", num_query_batches)
    for batch_num in progress_bar(range(num_query_batches), parent=master_bar):
        # Generate a batch of TSPs
        try:
            batch = next(dataset_query)
        except StopIteration:
            break

        # Convert batch to torch Variables
        x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
        x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
        x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
        x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
        y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
        y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)

        # Compute class weights (if uncomputed)
        if type(edge_cw) != torch.Tensor:
            edge_labels = y_edges.cpu().numpy().flatten()
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

        # Forward pass
        y_preds, loss = net_meta.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
        loss = loss.mean()  # Take mean of loss across multiple GPUs
        loss = loss / accumulation_steps  # Scale loss by accumulation steps
        # loss_for_meta += loss
        loss = loss  # /num_query_batches
        print("meta loss ", loss)
        loss.backward()

        # Compute error metrics and mean tour lengths
        # err_edges, err_tour, err_tsp, tour_err_idx, tsp_err_idx = edge_error(y_preds, y_edges, x_edges)
        pred_tour_len = mean_tour_len_edges(x_edges_values, y_preds)
        gt_tour_len = np.mean(batch.tour_len)

        # Update running data
        running_nb_data += batch_size
        running_loss += batch_size * loss.data.item() * accumulation_steps  # Re-scale loss
        running_pred_tour_len += batch_size * pred_tour_len
        running_gt_tour_len += batch_size * gt_tour_len
        running_nb_batch += 1

        # Log intermediate statistics
        result = ('loss:{loss:.4f} pred_tour_len:{pred_tour_len:.3f} gt_tour_len:{gt_tour_len:.3f}'.format(
            loss=running_loss / running_nb_data,
            pred_tour_len=running_pred_tour_len / running_nb_data,
            gt_tour_len=running_gt_tour_len / running_nb_data))
        master_bar.child.comment = result

    net_meta.load_state_dict(presever_old_weights_meta)

    # Compute statistics for full epoch
    loss = running_loss/ running_nb_data
    err_edges = 0 # running_err_edges/ running_nb_data
    err_tour = 0 # running_err_tour/ running_nb_data
    err_tsp = 0 # running_err_tsp/ running_nb_data
    pred_tour_len = running_pred_tour_len/ running_nb_data
    gt_tour_len = running_gt_tour_len/ running_nb_data

    return time.time()-start_epoch, loss, err_edges, err_tour, err_tsp, pred_tour_len, gt_tour_len


def metrics_to_str(epoch, time, learning_rate, loss, err_edges, err_tour, err_tsp, pred_tour_len, gt_tour_len):
    result = ( 'epoch:{epoch:0>2d}\t'
               'time:{time:.1f}h\t'
               'lr:{learning_rate:.2e}\t'
               'loss:{loss:.4f}\t'
               # 'err_edges:{err_edges:.2f}\t'
               # 'err_tour:{err_tour:.2f}\t'
               # 'err_tsp:{err_tsp:.2f}\t'
               'pred_tour_len:{pred_tour_len:.3f}\t'
               'gt_tour_len:{gt_tour_len:.3f}'.format(
                   epoch=epoch,
                   time=time/3600,
                   learning_rate=learning_rate,
                   loss=loss,
                   pred_tour_len=pred_tour_len,
                   gt_tour_len=gt_tour_len))
    return result



def final_evaluate(net_meta, config_meta, config_task, master_bar, mode='test'):

    net_task = copy.copy(net_meta)    
    net_task.eval()

    # Assign parameters
    num_nodes = config_task.num_nodes
    num_neighbors = config_task.num_neighbors
    batch_size = config_meta.batch_size
    beam_size = config_meta.beam_size
    val_filepath = config_task.val_filepath
    test_filepath = config_task.test_filepath

    # Load TSP data
    if mode == 'val':
        dataset = GoogleTSPReader(num_nodes, num_neighbors, batch_size=batch_size, filepath=val_filepath, split_details={'split_percentage':50,'split_id':1})
    elif mode == 'test':
        dataset = GoogleTSPReader(num_nodes, num_neighbors, batch_size=batch_size, filepath=test_filepath,  split_details={'split_percentage':90,'split_id':1})
    batches_per_epoch = dataset.max_iter

    # Convert dataset to iterable
    dataset = iter(dataset)
    
    # Initially set loss class weights as None
    edge_cw = None

    # Initialize running data
    running_loss = 0.0
    running_pred_tour_len = 0.0
    running_gt_tour_len = 0.0
    running_nb_data = 0
    running_nb_batch = 0
    running_opt_gap_sum = 0

    with torch.no_grad():
        start_test = time.time()
        print(" TEST BPE ", batches_per_epoch)
        for batch_num in progress_bar(range(batches_per_epoch), parent=master_bar):
            # Generate a batch of TSPs
            try:
                batch = next(dataset)
            except StopIteration:
                break

            # Convert batch to torch Variables
            x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
            x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
            x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
            x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
            y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
            y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)
            
            # Compute class weights (if uncomputed)
            if type(edge_cw) != torch.Tensor:
                edge_labels = y_edges.cpu().numpy().flatten()
                edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

            # Forward pass
            y_preds, loss = net_task.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
            loss = loss.mean()  # Take mean of loss across multiple GPUs

            # Compute error metrics

            # Get batch beamsearch tour prediction
            if mode == 'val':  # Validation: faster 'vanilla' beamsearch
                bs_nodes = beamsearch_tour_nodes(
                    y_preds, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='logits')
            elif mode == 'test':  # Testing: beamsearch with shortest tour heuristic 
                bs_nodes = beamsearch_tour_nodes_shortest(
                    y_preds, x_edges_values, beam_size, batch_size, num_nodes, dtypeFloat, dtypeLong, probs_type='logits')
            
            # Compute mean tour length
            pred_tour_len, all_tour_lens = mean_tour_len_nodes(x_edges_values, bs_nodes,  return_all_tour_lens=True)
            gt_tour_len = np.mean(batch.tour_len)

            opt_gap_sample_wise = np.divide(np.subtract(all_tour_lens, batch.tour_len), batch.tour_len)  #
            running_opt_gap_sum += opt_gap_sample_wise.sum()


            # Update running data
            running_nb_data += batch_size
            running_loss += batch_size* loss.data.item()
            running_pred_tour_len += batch_size* pred_tour_len
            running_gt_tour_len += batch_size* gt_tour_len
            running_nb_batch += 1

            # Log intermediate statistics
            result = ('loss:{loss:.4f} pred_tour_len:{pred_tour_len:.3f} gt_tour_len:{gt_tour_len:.3f}'.format(
                loss=running_loss/running_nb_data,
                pred_tour_len=running_pred_tour_len/running_nb_data,
                gt_tour_len=running_gt_tour_len/running_nb_data))
            master_bar.child.comment = result

    # Compute statistics for full epoch
    loss = running_loss/ running_nb_data
    err_edges = 0 # running_err_edges/ running_nb_data
    err_tour = 0 # running_err_tour/ running_nb_data
    err_tsp = 0 # running_err_tsp/ running_nb_data
    pred_tour_len = running_pred_tour_len/ running_nb_data
    gt_tour_len = running_gt_tour_len/ running_nb_data

    opt_gap_dataset = 100*running_opt_gap_sum.item()/running_nb_data

    return time.time()-start_test, loss, err_edges, err_tour, err_tsp, pred_tour_len, gt_tour_len, opt_gap_dataset



def test(net_meta, config_meta, config_task, master_bar, mode='test', num_fine_tune_steps=1):

    #FINE-TUNING

    print("time start fine tuning ", datetime.now())

    net_task = copy.copy(net_meta)
    net_task.train()
    optimizer_task = torch.optim.Adam(net_task.parameters(), lr=config_task.learning_rate)

    num_nodes = config_task.num_nodes
    num_neighbors = config_task.num_neighbors
    batch_size = config_meta.batch_size
    batches_per_epoch = config_meta.batches_per_epoch
    val_filepath = config_task.val_filepath
    accumulation_steps = config_task.accumulation_steps

    # Load TSP data

    if mode == 'val':
        dataset_support = GoogleTSPReader(num_nodes, num_neighbors, batch_size=batch_size, filepath=val_filepath,
                                  split_details={'split_percentage': 50, 'split_id': 0})


    elif mode == 'test':
        dataset_support = GoogleTSPReader(num_nodes, num_neighbors, batch_size=batch_size, filepath=config_task.train_filepath,
                                  split_details={'split_percentage': 100, 'split_id': 0}, shuffleDataset=False)



    print("Dataset support size", len(dataset_support))

    batches_per_epoch = num_fine_tune_steps
    print("batches per epoch", batches_per_epoch)
    # Convert dataset to iterable

    dataset_support = iter(dataset_support)


    # Initially set loss class weights as None
    edge_cw = None

    # Initialize running data
    running_loss = 0.0
    running_pred_tour_len = 0.0
    running_gt_tour_len = 0.0
    running_nb_data = 0
    running_nb_batch = 0


    print("num_fine_tune_steps ", num_fine_tune_steps)


    for batch_num in progress_bar(range(batches_per_epoch), parent=master_bar):
        # Generate a batch of TSPs
        try:
            batch = next(dataset_support)
        except StopIteration:
            break

        print("batch_num_fine_tune ", batch_num)


        if(batch_num >= num_fine_tune_steps):
            print(" done fine _tuning " )
            break


        # Convert batch to torch Variables
        x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
        x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
        x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
        x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
        y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)
        y_nodes = Variable(torch.LongTensor(batch.nodes_target).type(dtypeLong), requires_grad=False)

        # Compute class weights (if uncomputed)
        if type(edge_cw) != torch.Tensor:
            edge_labels = y_edges.cpu().numpy().flatten()
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

        # Forward pass
        y_preds, loss = net_task.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
        loss = loss.mean()  # Take mean of loss across multiple GPUs
        loss = loss / accumulation_steps  # Scale loss by accumulation steps
        loss.backward()

        # Backward pass
        if (batch_num + 1) % accumulation_steps == 0:
            optimizer_task.step()
            optimizer_task.zero_grad()

        # Compute error metrics and mean tour lengths
        pred_tour_len = mean_tour_len_edges(x_edges_values, y_preds)
        gt_tour_len = np.mean(batch.tour_len)

        # Update running data
        running_nb_data += batch_size
        running_loss += batch_size * loss.data.item() * accumulation_steps  # Re-scale loss
        running_pred_tour_len += batch_size * pred_tour_len
        running_gt_tour_len += batch_size * gt_tour_len
        running_nb_batch += 1

        # Log intermediate statistics
        result = ('loss:{loss:.4f} pred_tour_len:{pred_tour_len:.3f} gt_tour_len:{gt_tour_len:.3f}'.format(
            loss=running_loss / running_nb_data,
            pred_tour_len=running_pred_tour_len / running_nb_data,
            gt_tour_len=running_gt_tour_len / running_nb_data))
        master_bar.child.comment = result



        if(batch_num%50==0 and mode=='test'):
            val_time, val_loss, val_err_edges, val_err_tour, val_err_tsp, val_pred_tour_len, val_gt_tour_len, opt_gap_dataset = final_evaluate(
                        net_task,config_meta, config_task, master_bar, mode='test')

            print(" ********* at step ", batch_num, " OPT GAP is  ", opt_gap_dataset)


            print("time end fine tuning ", datetime.now())


    ##FINE TUNING END#
    if (mode == 'val'):
        val_time, val_loss, val_err_edges, val_err_tour, val_err_tsp, val_pred_tour_len, val_gt_tour_len, opt_gap_dataset = final_evaluate(
            net_task, config_meta, config_task, master_bar, mode='val')


    return val_time, val_loss, val_err_edges, val_err_tour, val_err_tsp, val_pred_tour_len, val_gt_tour_len, opt_gap_dataset



def main(config_meta):
    # Instantiate the network
    net_meta = nn.DataParallel(ResidualGatedGCNModel(config_meta, dtypeFloat, dtypeLong))
    if torch.cuda.is_available():
        print("cuda available")
    net_meta.cuda()
    print(net_meta)

    # Compute number of network parameters
    nb_param = 0
    for param in net_meta.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters:', nb_param)
 
    # Create log directory
    log_dir = f"./{config_meta.output_dir_name}/{config_meta.root_dir_name}/{config_meta.expt_name}/"
    os.makedirs(log_dir, exist_ok=True)
    json.dump(config_meta, open(f"{log_dir}/config_meta.json", "w"), indent=4)
    writer = SummaryWriter(log_dir)  # Define Tensorboard writer

    learning_rate = config_meta.learning_rate
    max_epochs = config_meta.max_epochs
    epoch_bar = master_bar(range(max_epochs))

    val_every = config_meta.val_every
    test_every = config_meta.test_every
    num_meta_steps = config_meta.num_meta_steps
    decay_rate = config_meta.decay_rate
    optimizer_meta = torch.optim.Adam(net_meta.parameters(), lr=learning_rate)
    print(optimizer_meta)

    val_loss_across_tasks_sum_old = 1e6

    best_opt_gap =10000000

    for epoch in epoch_bar:

        sys.stdout.flush()
        print("epoch ", epoch)
        for meta_steps in range(0,num_meta_steps ):

            print("meta step id ", meta_steps)

            for task in tasks_list:
                print("task ", task)
                config_task = dict_tasks_configs[str(task)]

                net_task = copy.copy(net_meta)

                writer.add_scalar('learning_rate', learning_rate, epoch)

                # Train
                train_time, train_loss, train_err_edges, train_err_tour, train_err_tsp, train_pred_tour_len, train_gt_tour_len = train_one_step(net_meta,
                    net_task, config_task, config_meta, epoch_bar)
                epoch_bar.write('t: ' + metrics_to_str(epoch, train_time, learning_rate, train_loss, train_err_edges,
                                                       train_err_tour, train_err_tsp, train_pred_tour_len,
                                                       train_gt_tour_len))
                writer.add_scalar('loss/train_loss', train_loss, epoch)
                writer.add_scalar('pred_tour_len/train_pred_tour_len', train_pred_tour_len, epoch)
                writer.add_scalar('optimality_gap/train_opt_gap', train_pred_tour_len / train_gt_tour_len - 1, epoch)


            print("meta updated")
            optimizer_meta.step()
            optimizer_meta.zero_grad()

        #VAL TEST PART#

        if epoch % val_every == 0 or epoch == max_epochs - 1:
            # Validate

            val_loss_across_tasks_sum =  0
            sum_opt_gap_of_val_dataset = 0

            for task in tasks_list:
                print("validate task ", task)
                config_task = dict_tasks_configs[str(task)]

                val_time, val_loss, val_err_edges, val_err_tour, val_err_tsp, val_pred_tour_len, val_gt_tour_len, opt_gap_dataset = test(
                    net_meta,config_meta, config_task, epoch_bar, mode='val', num_fine_tune_steps=config_meta.batches_per_epoch)

                sum_opt_gap_of_val_dataset+=opt_gap_dataset

                epoch_bar.write(
                    'v: ' + metrics_to_str(epoch, val_time, learning_rate, val_loss, val_err_edges, val_err_tour,
                                           val_err_tsp, val_pred_tour_len, val_gt_tour_len))

                val_loss_across_tasks_sum+=val_loss



                writer.add_scalar('loss/val_loss', val_loss, epoch)
                writer.add_scalar('pred_tour_len/val_pred_tour_len', val_pred_tour_len, epoch)
                writer.add_scalar('optimality_gap/val_opt_gap', val_pred_tour_len / val_gt_tour_len - 1, epoch)

                print("OPT GAP VAL ", opt_gap_dataset, " % ")

            # Save checkpoint
            if sum_opt_gap_of_val_dataset < best_opt_gap:
                best_opt_gap = sum_opt_gap_of_val_dataset  # Update best prediction
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net_meta.state_dict(),
                    'optimizer_state_dict': optimizer_meta.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, log_dir + "best_val_checkpoint.tar")

                print(" new best opt gap ", best_opt_gap)

            # Update learning rate
            if val_loss_across_tasks_sum > 0.99 * val_loss_across_tasks_sum_old:
                learning_rate /= decay_rate
                optimizer_meta = update_learning_rate(optimizer_meta, learning_rate)
                print("updated learning rate to ", learning_rate)

            val_loss_across_tasks_sum_old = val_loss_across_tasks_sum  # Update old validation loss



        if False and epoch % test_every == 0 or epoch == max_epochs - 1:

            all_tasks_list = tasks_lists_test + tasks_list
            for task in all_tasks_list:
                print("test task ", task)
                if task in tasks_lists_test:
                    config_task = dict_test_tasks_config[str(task)]
                else:
                    config_task = dict_tasks_configs[str(task)]

            # Test
                test_time, test_loss, test_err_edges, test_err_tour, test_err_tsp, test_pred_tour_len, test_gt_tour_len, opt_gap_dataset = test(
                    net_meta, config_meta, config_task, epoch_bar, mode='test')
                epoch_bar.write('T: ' + metrics_to_str(epoch, test_time, learning_rate, test_loss, test_err_edges,
                                                       test_err_tour, test_err_tsp, test_pred_tour_len,
                                                       test_gt_tour_len))
                writer.add_scalar('loss/test_loss', test_loss, epoch)

                writer.add_scalar('pred_tour_len/test_pred_tour_len', test_pred_tour_len, epoch)
                writer.add_scalar('optimality_gap/test_opt_gap', test_pred_tour_len / test_gt_tour_len - 1, epoch)
                print(" TEST PRED TOUR LENGTH  ", test_pred_tour_len)
                print(" TEST GT TOUR LENGTH ", test_gt_tour_len)
                print("OPT GAP TESTING DATASET ", opt_gap_dataset , " % ")

        # Save training checkpoint at the end of epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': net_meta.state_dict(),
            'optimizer_state_dict': optimizer_meta.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, log_dir + "last_train_checkpoint.tar")

        # Save checkpoint after every 250 epochs
        if epoch != 0 and (epoch % 250 == 0 or epoch == max_epochs - 1):
            torch.save({
                'epoch': epoch,
                'model_state_dict': net_meta.state_dict(),
                'optimizer_state_dict': optimizer_meta.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, log_dir + f"checkpoint_epoch{epoch}.tar")

    return net_meta


# In[ ]:


if TESTING==False:
    # del net
    net = main(config_meta)


if TESTING==True:
    print (" IN TESTING")
    
    #Instantiate the network
    net = nn.DataParallel(ResidualGatedGCNModel(config_meta, dtypeFloat, dtypeLong))
    net.cuda()
    print(net)

    # Compute number of network parameters
    nb_param = 0
    for param in net.parameters():
        nb_param += np.prod(list(param.data.size()))
    print('Number of parameters:', nb_param)
    
    # Define optimizer
    learning_rate = 0.0001#config_meta.learning_rate
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    print(optimizer)


    # Load checkpoint
    log_dir = f"./{config_meta.output_dir_name}/{config_meta.root_dir_name}/{config_meta.expt_name}/"
    if torch.cuda.is_available():
        checkpoint = torch.load(log_dir+"best_val_checkpoint.tar")
    else:
        checkpoint = torch.load(log_dir+"best_val_checkpoint.tar", map_location='cpu')
    # Load network state
    net.load_state_dict(checkpoint['model_state_dict'])
    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Load other training parameters
    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    val_loss = checkpoint['val_loss']
    for param_group in optimizer.param_groups:
        learning_rate = param_group['lr']
    print(f"Loaded checkpoint from epoch {epoch}")


#if notebook_mode==True:
if TESTING== True:
    print(" Evaluating the loaded model ")
    # Set evaluation mode
    net.eval()
    
    
    for task in tasks_lists_test:

        number_of_fst_steps_list =[1050]
        
        for num_fsteps in number_of_fst_steps_list:
    
            print(" EXPT For num_fsteps ", num_fsteps)
            print("test task ", task)
            if task in tasks_lists_test:
                config_task = dict_test_tasks_config[str(task)]
            else:
                config_task = dict_tasks_configs[str(task)]
    
            num_nodes = config_task.num_nodes
            num_neighbors = config_task.num_neighbors
            beam_size = config_meta.beam_size
            test_filepath = config_task.test_filepath

            epoch_bar = master_bar(range(epoch+1, epoch+2))

            for epoch in epoch_bar:
                config_task.val_filepath = config_task.test_filepath
                # Greedy search
                config_task.beam_size = 1
                config_task.learning_rate = 0.0001
                t=time.time()
                val_time, val_loss, val_err_edges, val_err_tour, val_err_tsp, val_pred_tour_len, val_gt_tour_len, opt_gap_dataset = test(
                        net,config_meta, config_task, epoch_bar, mode='test', num_fine_tune_steps=num_fsteps)
                print("G: Time: {}s".format(time.time() - t))
                epoch_bar.write(
                    'G: ' + metrics_to_str(epoch, val_time, learning_rate, val_loss, val_err_edges, val_err_tour,
                                           val_err_tsp, val_pred_tour_len, val_gt_tour_len))

                print(" Num of fine tuning steps done """, num_fsteps)
                print(" Opt gap "" ", opt_gap_dataset)