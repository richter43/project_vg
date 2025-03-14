import multiprocessing
import test
from os.path import join, exists
from datetime import datetime
import math
import logging
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from torch import nn
import pickle
from network import sos_loss


import localparser as parser
import util
import commons
import network
import datasets_ws

torch.backends.cudnn.benchmark = True  # Provides a speedup

# %% Initial setup: parser, logging...

# Parser, ignore
args = parser.parse_arguments()
# Time for logging purposes
start_time = datetime.now()

if args.use_mega == "y":
    util.init_mega(args)

#### Initialize cloud
if args.load_from == "":
    # Defining the output folder taking into consideration the "running folder" argument
    # (Is storing it into the parsed args namespace a standard procedure?)
    args.output_folder = join("runs", args.exp_name, start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    if args.use_mega == "y":
        # create the folder in the cloud, it will store your model
        args.m.create_folder(args.output_folder)
        args.mega_folder = util.MyFind(args.m,args.output_folder)
else:
    assert args.use_mega == "y"
    args.mega_folder = util.MyFind(args.m, args.load_from)
    assert args.mega_folder != None

    logging.info(f"Resuming training starting from checkpoint in {args.load_from}")
    args.output_folder = args.load_from


# Logging setup - self-explanatory
commons.setup_logging(args.output_folder)
# Making computation deterministic given a predetermined seed - self-explanatory
commons.make_deterministic(args.seed)

logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.output_folder}")
logging.info(
    f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

# %% Creation of Datasets

# TripletsDataset is a child class of BaseDataset, the difference is that TD is especialized
# for training using the triplet loss

logging.debug(f"Loading dataset Pitts30k from folder {args.datasets_folder}")

val_ds = datasets_ws.BaseDataset(
    args, args.datasets_folder, args.dataset, "val")
logging.info(f"Val set: {val_ds}")

test_ds = datasets_ws.BaseDataset(
    args, args.datasets_folder, args.dataset, "test")
logging.info(f"Test set: {test_ds}")

# %% Initialize model
if args.layer == "solar":
    model = network.GeoLocalizationNetSOA(args)
else:
    model = network.GeoLocalizationNet(args)

# Loading initial cluster values
if exists(args.ancillaries_file) and args.layer == "net" and args.load_from == "":
    ancillaries = torch.load(args.ancillaries_file)
    centroids = ancillaries["centroids"]
    traindesc = ancillaries["traindesc"]
    
    model.aggregation.init_params(centroids, traindesc)

# Loading pre-trained state dicts
if args.load_from != "":
    if args.layer == "net":
        model.aggregation.conv.bias = None
    logging.info(f"Loading previous model from cloud")
    util.init_tmp_dir(args)
    args.checkpoint = torch.load(join(args.output_folder, "last_model.pth"))
    model.load_state_dict(args.checkpoint['model_state_dict'])
    
    

model = model.to(args.device)

# %% Setup Optimizer and Loss
if args.optim.lower() == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
elif args.optim.lower() == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

# Metric used in ML in order to minimize the distance to a positive example and maximize the distance
# to a negative example given an input value
criterion_triplet = nn.TripletMarginLoss(
    margin=args.margin, p=2, reduction="sum")

best_r5 = 0  # Recall after 5 recommendations
not_improved_num = 0
starting_epoch = 0

#### Loading model
if args.load_from != "":
    optimizer.load_state_dict(args.checkpoint['optimizer_state_dict'])
    starting_epoch = args.checkpoint['epoch_num'] + 1 
    #loss = checkpoint['loss']
    best_r5 = args.checkpoint['best_r5']
    not_improved_num = args.checkpoint['not_improved_num']
    args.lr = args.checkpoint['lr']
    args.train_positives_dist_threshold = args.checkpoint['train_positives_dist_threshold']
    # Added loading serialized contents of the data object
    triplets_ds = pickle.loads(args.checkpoint['ds_state']) #TODO: May need to change the location of the files
else:
    triplets_ds = datasets_ws.TripletsDataset(
        args, args.datasets_folder, args.dataset, "train", args.negs_num_per_query)
    
logging.info(f"Train query set: {triplets_ds}")
    
    
logging.info(f"Output dimension of the model is {args.features_dim}")

if starting_epoch != 0:
    #Restoring numpy random state
    np.random.set_state(args.checkpoint['np_random_state'])
    logging.info(f"starting epoch is not zero, iterating through dataloader")

# %% Training loop
for epoch_num in range(starting_epoch, args.epochs_num):
    #skipping epochs when resuming: we still need to iterate through the dataloader, otherwise when resuming
    #we will train with the same data we used during the very first training session
    # if epoch_num>=starting_epoch:
    #     logging.info(f"Start training epoch: {epoch_num:02d}")
    # else:
    #     logging.info(f"database iteration is {epoch_num}")
    logging.info(f"Start training epoch: {epoch_num:02d}")
    epoch_start_time = datetime.now()
    epoch_losses = np.zeros((0, 1), dtype=np.float32)

    # How many loops should an epoch last (default is 5000/1000=5)
    loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
    for loop_num in range(loops_num):
        logging.debug(f"Cache: {loop_num} / {loops_num}")

        
        # Compute triplets to use in the triplet loss
        triplets_ds.is_inference = True
        triplets_ds.compute_triplets(args, model)
        triplets_ds.is_inference = False

        # Transform to DataLoader
        triplets_dl = DataLoader(dataset=triplets_ds, num_workers=args.num_workers,
                                 batch_size=args.train_batch_size,
                                 collate_fn=datasets_ws.collate_fn,
                                 pin_memory=(args.device == "cuda"),
                                 drop_last=True)
        
        model = model.train()
        
        # images shape: (train_batch_size*12)*3*H*W ; by default train_batch_size=4, H=480, W=640
        # triplets_local_indexes shape: (train_batch_size*10)*3 ; because 10 triplets per query
        for images, triplets_local_indexes, _ in tqdm(triplets_dl, ncols=100):
            # Compute features of all images (images contains queries, positives and negatives)
            features = model(images.to(args.device))
            # This is implicitly casted to a tensor afterwards
            loss_triplet = 0
            loss_sos = 0
            # View
            triplets_local_indexes = torch.transpose(
                triplets_local_indexes.view(args.train_batch_size, args.negs_num_per_query, 3), 1, 0)
            for triplets in triplets_local_indexes:
                queries_indexes, positives_indexes, negatives_indexes = triplets.T
                
                loss_triplet += criterion_triplet(features[queries_indexes],
                                                  features[positives_indexes],
                                                  features[negatives_indexes])
                if args.layer == "solar":
                    loss_sos +=sos_loss(features[queries_indexes], features[positives_indexes], features[negatives_indexes])
            del features
            if args.layer == "solar":
              loss_triplet = loss_triplet + args.sos_lambda* torch.pow(loss_sos,0.5)
            loss_triplet /= (args.train_batch_size * args.negs_num_per_query)

            # set_to_none=True local optimization does not translate to global time optimization
            optimizer.zero_grad()
            loss_triplet.backward()
            optimizer.step()

            # Keep track of all losses by appending them to epoch_losses
            batch_loss = loss_triplet.item()
            epoch_losses = np.append(epoch_losses, batch_loss)
            del loss_triplet
        
        
        logging.debug(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
                        f"current batch triplet loss = {batch_loss:.4f}, " +
                        f"average epoch triplet loss = {epoch_losses.mean():.4f}")
        
    # if epoch_num < starting_epoch:
    #     #we're iterating through dataloader, no need to compute recalls (most likely they're in the log files for the given epoch_num)
    #     continue

    logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                f"average epoch triplet loss = {epoch_losses.mean():.4f}")
    
    # Compute recalls on validation set
    recalls, recalls_str = test.test(args, val_ds, model)
    logging.info(f"Recalls on val set {val_ds}: {recalls_str}")

    is_best = recalls[1] > best_r5

    # Save checkpoint, which contains all training parameters
    ## np_random_state, used for restoring the numpy random state, preserves determinism
    util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls, "best_r5": recalls[1] if is_best else best_r5,
                        "not_improved_num": not_improved_num, "lr" : args.lr, "train_positives_dist_threshold": args.train_positives_dist_threshold,
                        "np_random_state": np.random.get_state(), "ds_state": pickle.dumps(triplets_ds)}, is_best, filename="last_model.pth")

    if args.use_mega == "y":
        #upload to mega. Done on a different thread. I'm assuming this is a much faster operation than a training epoch
        util.upload_checkpoint(args, is_best)
    
    # If recall@5 did not improve for "many" epochs, stop training
    if is_best:
        logging.info(
            f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}")
        best_r5 = recalls[1]
        not_improved_num = 0
    else:
        not_improved_num += 1
        logging.info(
            f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}")
        if not_improved_num >= args.patience:
            logging.info(
                f"Performance did not improve for {not_improved_num} epochs. Stop training.")
            break


logging.info(f"Best R@5: {best_r5:.1f}")
logging.info(
    f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

# Test best model on test set
best_model_state_dict = torch.load(join(args.output_folder, "best_model.pth"))[
    "model_state_dict"]
model.load_state_dict(best_model_state_dict)

recalls, recalls_str = test.test(args, test_ds, model)
logging.info(f"Recalls on {test_ds}: {recalls_str}")
