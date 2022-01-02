
import argparse


def parse_arguments():

    # %% Original arguments

    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=4,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--epochs_num", type=int, default=50,
                        help="Maximum number of epochs to train for")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float,
                        default=0.00001, help="Learning rate")
    parser.add_argument("--cache_refresh_rate", type=int, default=1000,
                        help="How often to refresh cache, in number of queries")
    parser.add_argument("--queries_per_epoch", type=int, default=5000,
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument("--negs_num_per_query", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--neg_samples_num", type=int, default=1000,
                        help="How many negatives to use to compute the hardest ones")
    # Other parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str,
                        default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=8,
                        help="num_workers for all dataloaders")
    parser.add_argument("--val_positive_dist_threshold", type=int,
                        default=25, help="Val/test threshold in meters")
    parser.add_argument("--train_positives_dist_threshold",
                        type=int, default=10, help="Train threshold in meters")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str,
                        required=True, help="Path with datasets")
    parser.add_argument("--exp_name", type=str, default="default",
                        help="Folder name of the current run (saved in ./runs/)")
    
    # Layer to be used and related parameters
    # %% New arguments
    parser.add_argument("--layer", type=str, default="avg", choices=["avg", "net", "gem"],
                        help="Model frontend used (E.g: \"avg\", \"net\")")
    parser.add_argument("--num_clusters", type =int, default = 64, 
                        help = "Number of clusters used in the NetVLAD layer")
    parser.add_argument("--load_from", type = str, default = "", 
                        help = "Folder (in cloud) of the model to be loaded. Resumes training from checkpoint")
    parser.add_argument("--use_mega", type = str, default = "y", 
                        help = "Load/store your result into the mega cloud")
    parser.add_argument("--mega_username", type = str, default = "c.blangio@gmail.com", 
                        help = "Username of the owner of the Mega cloud you want to access for loading a model")
    parser.add_argument("--pk", type=int, default=3, 
                        help="Exponent of GeM pooling")
    parser.add_argument("--minval", type=float, default=1e-6, 
                        help="Miniminal value that is allowed in order to avoid computation skewing (i.e: dividing by zero)")

    args = parser.parse_args()

    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")
    return args
