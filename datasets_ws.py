
import os
import faiss
import logging
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
from os.path import join
import torch
import time
from multiprocessing import Pool

import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from torch.utils.data.dataloader import DataLoader
from sklearn.neighbors import NearestNeighbors

from torch import nn
from argparse import Namespace

# %% Typing for improving readability
from typing import Type, Tuple
DatasetFullLength = int
FinalConvLayerSize = int
TripletImages = torch.Tensor
TripletLocalIndexes = torch.Tensor
TripletGlobalIndexes = torch.Tensor

# %% Code

base_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

bright_t = transforms.ColorJitter(brightness=[1,2])
contrast_t = transforms.ColorJitter(contrast = [2,5])
saturation_t = transforms.ColorJitter(saturation = [1,3])
hue_t = transforms.ColorJitter(hue = 0.2)
gs_t = transforms.Grayscale(3)
hflip_t = transforms.RandomHorizontalFlip(p = 1)
rp_t = transforms.RandomPerspective(p = 1, distortion_scale = 0.5)
rot_t = transforms.RandomRotation(degrees = 90)
rndcrop_t = transforms.RandomCrop(128)

aug_transformations = {
    "CS-HF": transforms.Compose([contrast_t, saturation_t, hflip_t]),
    "H-RP": transforms.Compose([hue_t, rp_t]),
    "B-GS-R": transforms.Compose([bright_t, gs_t, rot_t])
    "RC": transforms.Compose([rndcrop_t])
    "GS": transforms.Compose([gs_t])
    "F-R": transforms.Compose([hflip_t, rot_t])
    "n": None
    }


"""
aug_transform = transforms.Compose([
    transforms.RandomCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
"""

def path_to_pil_img(path):
    return Image.open(path).convert("RGB")


def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (images,
        triplets_local_indexes, triplets_global_indexes).
        triplets_local_indexes are the indexes referring to each triplet within images.
        triplets_global_indexes are the global indexes of each image.
    Args:
        batch: list of tuple (images, triplets_local_indexes, triplets_global_indexes).
            considering each query to have 10 negatives (negs_num_per_query=10):
            - images: torch tensor of shape (12, 3, h, w).
            - triplets_local_indexes: torch tensor of shape (10, 3).
            - triplets_global_indexes: torch tensor of shape (12).
    Returns:
        images: torch tensor of shape (batch_size*12, 3, h, w).
        triplets_local_indexes: torch tensor of shape (batch_size*10, 3).
        triplets_global_indexes: torch tensor of shape (batch_size, 12).
    """
    images = torch.cat([e[0] for e in batch])
    triplets_local_indexes = torch.cat([e[1][None] for e in batch])
    triplets_global_indexes = torch.cat([e[2][None] for e in batch])
    for i, (local_indexes, global_indexes) in enumerate(zip(triplets_local_indexes, triplets_global_indexes)):
        # Increment local indexes by offset (len(global_indexes) is 12)
        local_indexes += len(global_indexes) * i
    return images, torch.cat(tuple(triplets_local_indexes)), triplets_global_indexes


class BaseDataset(data.Dataset):
    """Dataset with images from database and queries, used for inference (testing and building cache).
    """

    def __init__(self, args: Namespace, datasets_folder="datasets", dataset_name="pitts30k", split="train"):
        super().__init__()
        self.args = args
        self.dataset_name = dataset_name
        self.dataset_folder = join(
            datasets_folder, dataset_name, "images", split)
        self.split = split
        
        data_transform = aug_transformations[args.data_aug]
        self.aug_transform = transforms.Compose([
            transforms.RandomApply([data_transform], p = args.aug_prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
])

        # Adding the data augmentation boolean
        self.data_aug = False if args.data_aug == 'n' else True
        
        if not os.path.exists(self.dataset_folder):
            raise FileNotFoundError(
                f"Folder {self.dataset_folder} does not exist")

        # Read paths and UTM coordinates for all images.
        database_folder = join(self.dataset_folder, "database")
        queries_folder = join(self.dataset_folder, "queries")
        if not os.path.exists(database_folder):
            raise FileNotFoundError(f"Folder {database_folder} does not exist")
        if not os.path.exists(queries_folder):
            raise FileNotFoundError(f"Folder {queries_folder} does not exist")
        self.database_paths = sorted(
            glob(join(database_folder, "**", "*.jpg"), recursive=True))
        self.queries_paths = sorted(
            glob(join(queries_folder, "**", "*.jpg"),  recursive=True))
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split(
            "@")[1], path.split("@")[2]) for path in self.database_paths]).astype(np.float)
        self.queries_utms = np.array([(path.split(
            "@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(np.float)

        # Find soft_positives_per_query, which are within val_positive_dist_threshold (default 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        # Array of elements' indices close up to args.val_positive_dist_threshold given a query
        self.soft_positives_per_query = knn.radius_neighbors(self.queries_utms,
                                                             radius=args.val_positive_dist_threshold,
                                                             return_distance=False)

        self.images_paths = list(self.database_paths) + \
            list(self.queries_paths)

        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

    def __getitem__(self, index):
        img = path_to_pil_img(self.images_paths[index])
        img = base_transform(img)
        return img, index

    def __len__(self):
        return len(self.images_paths)

    def __repr__(self):
        return (f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >")

    def get_positives(self):
        return self.soft_positives_per_query

# Scrapped idea, memory usage is too high
# def faiss_range_search(index: faiss.IndexFlatL2, query_batch: np.ndarray, args: Namespace) -> Tuple[np.ndarray, np.ndarray]:
#     return index.range_search(query_batch, args.train_positives_dist_threshold)


class TripletsDataset(BaseDataset):
    """Dataset used for training, it is used to compute the triplets
    with TripletsDataset.compute_triplets().
    If is_inference == True, uses methods of the parent class BaseDataset,
    this is used for example when computing the cache, because it requires
    computing features of each image, not triplets.
    """

    def __init__(self, args: Namespace, datasets_folder="datasets", dataset_name="pitts30k", split="train", negs_num_per_query=10):

        super().__init__(args, datasets_folder, dataset_name, split)
        # Number of negatives to randomly sample
        self.neg_samples_num = args.neg_samples_num
        # Number of negatives per query in each batch
        self.negs_num_per_query = negs_num_per_query
        self.neg_cache = [np.empty((0,), dtype=np.int32)
                          for _ in range(self.queries_num)]
        self.is_inference = False

        # Find hard_positives_per_query, which are within train_positives_dist_threshold (10 meters)
        # Report: faiss CAN'T  be used here due to an obscure decision for batch computation
        # (Using IndexFlatL2 with an input's shape larger than 20 changes how the computation is done (Not euclidean distance) which, for our purpose, ends up with a distance vector filled with zeroes)
        # Note: there's a lot of repeated vectors, how could this be improved?
        # Doesn't seem to be worth it, this is computed only once

        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.hard_positives_per_query = list(knn.radius_neighbors(self.queries_utms,
                                             radius=args.train_positives_dist_threshold,  # 10 meters
                                             return_distance=False))

        # Some queries might have no positive, we should remove those queries.
        queries_without_any_hard_positive = np.where(np.array(
            [len(p) for p in self.hard_positives_per_query], dtype=object) == 0)[0]
        if len(queries_without_any_hard_positive) != 0:
            logging.info(f"There are {len(queries_without_any_hard_positive)} queries without any positives " +
                         "within the training set. They won't be considered as they're useless for training.")
        # Remove queries without positives
        self.hard_positives_per_query = np.delete(
            self.hard_positives_per_query, queries_without_any_hard_positive)
        self.queries_paths = np.delete(
            self.queries_paths, queries_without_any_hard_positive)

        # Recompute images_paths and queries_num because some queries might have been removed
        self.images_paths = list(self.database_paths) + \
            list(self.queries_paths)
        self.queries_num = len(self.queries_paths)

    def __getitem__(self, index: int) -> Tuple[TripletImages, TripletLocalIndexes, TripletGlobalIndexes]:
        
        aug_transform = self.aug_transform
        if self.is_inference:
            # At inference time return the single image. This is used for caching
            return super().__getitem__(index)

        # triplets_global_indexes is obtained after externally calling the compute_triplets method
        # Extraction of useful values

        query_index, best_positive_index, neg_indexes = torch.split(
            self.triplets_global_indexes[index],
            (1, 1, self.negs_num_per_query))

        # The following commands take the indexes of the respective image,
        # reads from disk, transforms to tensor and normalizes according to Imagenet
        # parameters
        
        # It also augments the dataset if chosen to do so
        if self.split == "train" and self.data_aug:
            query = aug_transform(path_to_pil_img(
                self.queries_paths[query_index]))
    
            positive = aug_transform(path_to_pil_img(
                self.database_paths[best_positive_index]))
    
            negatives = [aug_transform(path_to_pil_img(
                self.database_paths[i])) for i in neg_indexes]
        else:
            query = base_transform(path_to_pil_img(
                self.queries_paths[query_index]))
    
            positive = base_transform(path_to_pil_img(
                self.database_paths[best_positive_index]))
    
            negatives = [base_transform(path_to_pil_img(
                self.database_paths[i])) for i in neg_indexes]

        # Stacking actual images' data into a tensor
        images = torch.stack((query, positive, *negatives), 0)
        triplets_local_indexes = torch.empty((0, 3), dtype=torch.int)

        # Creation of a local index list
        # A local index is just a list of tuples that contain the indexes of the images
        # relative to the given set of images (for instance, given a single set of triplet images
        # the indexes are query_image=0 pos_image->1 neg_image->[2,11])

        for neg_num in range(len(neg_indexes)):
            triplets_local_indexes = torch.cat(
                (triplets_local_indexes, torch.tensor([0, 1, 2+neg_num]).reshape(1, 3)))
        return images, triplets_local_indexes, self.triplets_global_indexes[index]

    def __len__(self) -> int:
        if self.is_inference:
            # At inference time return the number of images. This is used for caching
            return super().__len__()
        else:
            return len(self.triplets_global_indexes)

    @staticmethod
    def compute_cache(args: Namespace, model: Type[nn.Module], subset_ds: Subset, cache_shape: Tuple[DatasetFullLength, FinalConvLayerSize]) -> np.ndarray:
        """Compute the cache containing features of images, which is used to
        find best positive and hardest negatives."""
        subset_dl = DataLoader(dataset=subset_ds, num_workers=args.num_workers,
                               batch_size=args.infer_batch_size, shuffle=False,
                               pin_memory=(args.device == "cuda"))
        model = model.eval()

        cache = np.zeros(cache_shape, dtype=np.float32)
        with torch.no_grad():
            for images, indexes in tqdm(subset_dl, ncols=100):
                images = images.to(args.device)
                features = model(images)

                cache[indexes.numpy()] = features.cpu().numpy()
        return cache

    def get_query_features(self, query_index: int, cache: np.ndarray) -> np.ndarray:
        query_features = cache[query_index + self.database_num]
        if query_features is None:
            raise RuntimeError(f"For query {self.queries_paths[query_index]} " +
                               f"with index {query_index} features have not been computed!\n" +
                               "There might be some bug with caching")
        return query_features

    def get_best_positive_index(self, args: Namespace, query_index: int, cache: np.ndarray, query_features: np.ndarray) -> int:
        positives_features = cache[self.hard_positives_per_query[query_index]]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(positives_features)
        # Search the best positive (within 10 meters AND nearest in features space)
        _, best_positive_num = faiss_index.search(
            query_features.reshape(1, -1), 1)
        best_positive_index = self.hard_positives_per_query[query_index][best_positive_num[0]].item(
        )
        return best_positive_index

    def get_hardest_negatives_indexes(self, args, cache, query_features, neg_samples):
        neg_features = cache[neg_samples]
        faiss_index = faiss.IndexFlatL2(args.features_dim)
        faiss_index.add(neg_features)
        # Search the 10 nearest negatives (further than 25 meters and nearest in features space)
        _, neg_nums = faiss_index.search(
            query_features.reshape(1, -1), self.negs_num_per_query)
        neg_nums = neg_nums.reshape(-1)
        neg_indexes = neg_samples[neg_nums].astype(np.int32)
        return neg_indexes

    def compute_triplets(self, args: Namespace, model: Type[nn.Module]) -> None:
        """
        Creates the triplets_global_indexes tensor for the current dataset, which is a
        list of tuples containing the index of a given query with its best positive samples' indexes
        and a list of negative samples' indexes. (query_index, best_positive_index, *neg_indexes)
        Also computes the cache of all features for accelerating their access.

        Parameters
        ----------
        args : Namespace
            Argument namespace that contains all of the various settings useful at runtime.
        model : Type[nn.Module]
            Model which is going to be trained.
        """

        # Refer to datasets_ws.TripletsDataset documentation for more information on is_inference
        self.is_inference = True
        self.triplets_global_indexes = []
        # Take 1000 random queries
        sampled_queries_indexes = np.random.choice(
            self.queries_num, args.cache_refresh_rate, replace=False)
        # Take all database indexes
        database_indexes = list(range(self.database_num))
        #  Compute features for all images and store them in cache
        subset_ds = Subset(self, database_indexes +
                           list(sampled_queries_indexes + self.database_num))

        cache = self.compute_cache(
            args, model, subset_ds, (len(self), args.features_dim))

        # This loop's iterations could be done individually in the __getitem__(). This way is slower but clearer (and yields same results)
        for query_index in tqdm(sampled_queries_indexes, ncols=100):
            query_features = self.get_query_features(query_index, cache)
            best_positive_index = self.get_best_positive_index(
                args, query_index, cache, query_features)
            # Choose 1000 random database images (neg_indexes)
            # Question: How can we be certain that choosing images randomly results in a negative sample?
            neg_indexes = np.random.choice(
                self.database_num, self.neg_samples_num, replace=False)
            # Remove the eventual soft_positives from neg_indexes
            soft_positives = self.soft_positives_per_query[query_index]
            neg_indexes = np.setdiff1d(
                neg_indexes, soft_positives, assume_unique=True)
            # Concatenate neg_indexes with the previous top 10 negatives (neg_cache)
            neg_indexes = np.unique(np.concatenate(
                [self.neg_cache[query_index], neg_indexes]))
            # Search the hardest negatives
            neg_indexes = self.get_hardest_negatives_indexes(
                args, cache, query_features, neg_indexes)
            # Update nearest negatives in neg_cache
            self.neg_cache[query_index] = neg_indexes
            self.triplets_global_indexes.append(
                (query_index, best_positive_index, *neg_indexes))
        # self.triplets_global_indexes is a tensor of shape [1000, 12]
        self.triplets_global_indexes = torch.tensor(
            self.triplets_global_indexes)
