
import faiss
import torch
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
import localparser as parser
import util
import commons
import datasets_ws
import network
import multiprocessing
from os.path import join,exists
from datasets_ws import aug_transformations

def test(args, eval_ds, model):
    """Compute features of the given dataset and compute the recalls."""
    model = model.eval()
    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")
        # For database use "hard_resize", although it usually has no effect because database images have same resolution
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        
        all_features = np.empty((len(eval_ds), args.features_dim), dtype="float32")
        for inputs, indices in tqdm(database_dataloader, ncols=100):
            inputs = inputs.to(args.device)
            if args.data_aug == "R-I" or args.data_aug == "R-D":
              inputs = aug_transformations[args.data_aug](inputs)
            features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            all_features[indices.numpy(), :] = features
        
        logging.debug("Extracting queries features for evaluation/testing")
        queries_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num, eval_ds.database_num+eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=args.infer_batch_size, pin_memory=(args.device=="cuda"))
        for inputs, indices in tqdm(queries_dataloader, ncols=100):
            features = model(inputs.to(args.device))
            features = features.cpu().numpy()
            all_features[indices.numpy(), :] = features
    
    queries_features = all_features[eval_ds.database_num:]
    database_features = all_features[:eval_ds.database_num]
    
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    del database_features, all_features
    
    logging.debug("Calculating recalls")
    distances, predictions = faiss_index.search(queries_features, max(args.recall_values))
    
    #### For each query, check if the predictions are correct
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    return recalls, recalls_str

if __name__ == "__main__":
    # Parser, ignore
    args = parser.parse_arguments()
    
    #for loading model from cloud
    if args.use_mega == "y":
        assert args.load_from != "" #if you're using mega, you have to specify the folder in the cloud where model is stored
        assert args.model_folder == "" #if you're using mega it is not necessary to specify the local folder of the model (it will be downloaded)
        assert args.load_from != "" #this is where you should specify the cloud folder containing the model
        util.init_mega(args)
        args.mega_folder = util.MyFind(args.m, args.load_from)
        assert args.mega_folder != None #checking that the folder exists in cloud
        args.output_folder = args.load_from #just for compatibility with util functions
    else:
        assert args.model_folder != "" #you need to specify the folder where the model is stored
        args.output_folder = args.model_folder #compatibility with util functions
    
    # Logging setup - self-explanatory
    commons.setup_logging(args.output_folder)
    # Making computation deterministic given a predetermined seed - self-explanatory
    commons.make_deterministic(args.seed)

    logging.info(f"Arguments: {args}")
    logging.info(f"The model is being loaded from {args.output_folder}")
    logging.info(
        f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

    logging.debug(f"Loading dataset from folder {args.datasets_folder}")

    
    test_ds = datasets_ws.BaseDataset(
        args, args.datasets_folder, args.dataset, "test")
    logging.info(f"Test set: {test_ds}")
    
    # %% Initialize model
    if args.layer != "solar":
      model = network.GeoLocalizationNet(args)
    else:
      model = network.GeoLocalizationNetSOA(args)
    
    if args.layer == "net":
        model.aggregation.conv.bias = None
    
    # Loading pre-trained state dicts
    if args.load_from != "":
        logging.info(f"Loading previous model from cloud")
        util.init_tmp_dir(args)
    
    args.checkpoint = torch.load(join(args.output_folder, "best_model.pth"))
    model.load_state_dict(args.checkpoint['model_state_dict'])
    
    # Loading initial cluster values
    if exists(args.ancillaries_file) and args.layer == "net":
        ancillaries = torch.load(args.ancillaries_file)
        centroids = ancillaries["centroids"]
        traindesc = ancillaries["traindesc"]

        model.aggregation.init_params(centroids, traindesc)

    model = model.to(args.device)
    recalls, recalls_str = test(args, test_ds, model)
    logging.info(f"Recalls on {test_ds}: {recalls_str}")




