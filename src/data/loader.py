from src.data.helper import get_paired_image_ds, get_count_image_ds
from tensorflow.data import AUTOTUNE
from typing import List, Tuple
import tensorflow as tf


class RankDataIterator(object):
    """
    Generates a dataset of paired images with a label of 1. The datasset has the structure ((im1, im2), rank)
    Args:
        dataset: list of dictionaries with keys "source_image", "target_image", which are the relative paths to the image files within a directory
        path: path to the directory containing the image files
        size: size of the images
    """
    def __init__(
            self, 
            dataset: List[dict], 
            path: str, 
            size: Tuple[int, ...]):
        self.src_f = [path + d["source_image"] for d in dataset]
        self.tar_f = [path + d["target_image"] for d in dataset]
        self.size = size
        
    def build(
            self,  
            batch_size: int, 
            drop_remainder: bool, 
            shuffle: bool) -> tf.data.Dataset:
        rank_ds = get_paired_image_ds(self.src_f, self.tar_f, self.size, shuffle).batch(batch_size, drop_remainder=drop_remainder)
        return rank_ds.prefetch(AUTOTUNE)
    
class CountDataIterator(object):
    """
    Generates a dataset of images with a label of the object count. The dataset has the structure (im, count)
    Args:
        dataset: list of dictionaries with keys "image", "count", which are the relative paths to the image files within a directory and the object count within the image
        path: path to the directory containing the image files
        size: size of the images
    """
    def __init__(
            self, 
            dataset: List[dict], 
            path: str, 
            size: Tuple[int, ...]):
        self.images = [path + d["image"] for d in dataset]
        self.counts = [d["count"] for d in dataset]
        self.size = size

    def build(
            self, 
            batch_size: int, 
            drop_remainder: bool, 
            shuffle: bool) -> tf.data.Dataset:
        count_ds = get_count_image_ds(self.images, self.counts, self.size, shuffle).batch(batch_size, drop_remainder=drop_remainder)
        return count_ds.prefetch(AUTOTUNE)