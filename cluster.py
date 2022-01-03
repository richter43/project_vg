#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 23:37:30 2021

@author: foxtrot
"""

import os
from os.path import join, exists
from datetime import datetime
import logging
from argparse import Namespace

import torch
from torch.utils.data import Subset
import faiss

from network import GeoLocalizationNet
from localparser import parse_arguments
from datasets_ws import BaseDataset, TripletsDataset
import commons


def setup(args: Namespace) -> datetime:
    """
    Setting initial parameters relevant to debugging and logging
    """

    start_time = datetime.now()
    # Defining the output folder taking into consideration the "running folder" argument
    args.output_folder = join("runs", "cluster",
                              start_time.strftime('%Y-%m-%d_%H-%M-%S'))
    # Logging setup - self-explanatory
    commons.setup_logging(args.output_folder)
    # Making computation deterministic given a predetermined seed - self-explanatory
    commons.make_deterministic(args.seed)

    logging.info(f"Computing {args.clusters} centroids")

    extra_folder = join(".", "extra")

    if not exists(extra_folder):
        os.makedirs(extra_folder)

    return start_time


if __name__ == "__main__":

    args = parse_arguments()
    # The dataset of the relevant images is used (train, val or test, not the WHOLE dataset)

    start_time = setup(args)

    # Using the already existing BaseDataset to load the dataset, the various parameters
    # are set through arguments at execution time
    dataset = BaseDataset(
        args, args.datasets_folder, args.dataset, args.split)

    # Taking only the database indexes since we are getting the clusters of the database set, not the query ones
    database_indexes = list(range(dataset.database_num))
    #  Compute features for all images and store them in cache
    subset_ds = Subset(dataset, database_indexes)

    # Getting the model in clustering mode
    model = GeoLocalizationNet(args, cluster=True)
    model = model.to(args.device)

    cache = TripletsDataset.compute_cache(
        args, model, subset_ds, ((dataset.database_num, args.features_dim)))

    # Finding the centroids
    kmeans = faiss.Kmeans(args.features_dim, args.clusters)
    kmeans.train(cache)

    # Saving centroids
    torch.save({"centroids": kmeans.centroids, "traindesc": cache},
               args.ancillaries_file)

    logging.debug(f"Elapsed time: {datetime.now() - start_time}")
