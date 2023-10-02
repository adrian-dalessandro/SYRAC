import tensorflow as tf
from tensorflow.data import Dataset

def make_paired_ranking_data(data):
    """
    takes a list of N examples with format {"image": str, "count": int} and returns a list of all
    possible pairs of images with format {"source_image": str, "target_image": str} where source_image has
    a higher count than target_image
    """
    paired_data = []
    for i in range(len(data)):
        for j in range(len(data)):
            if data[i]["count"] > data[j]["count"]:
                paired_data.append({"source_image": data[i]["image"], "target_image": data[j]["image"]})
    return paired_data
    

@tf.function
def parse_images(image_path, size):
    image_string = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[size[0], size[1]])
    return image

def get_paired_image_ds(im_src, im_tar, size, shuffle=False):
    """
    get_paired_image_ds is a function that takes two lists of image paths and 
    returns a dataset of paired images with a label of 1
    """
    src_ds = Dataset.from_tensor_slices(im_src)
    tar_ds = Dataset.from_tensor_slices(im_tar)
    rank_ds = Dataset.zip((src_ds, tar_ds))
    if shuffle:
        rank_ds = rank_ds.shuffle(6000)
    rank_ds = rank_ds.map(lambda x, y: ((parse_images(x, size), parse_images(y, size)), 1), num_parallel_calls=tf.data.AUTOTUNE)
    return rank_ds

def get_count_image_ds(img_src, counts, size, shuffle=False):
    """
    get_count_image_ds is a function that takes a list of image paths and 
    returns a dataset of paired images
    """
    src_ds = Dataset.from_tensor_slices(img_src)
    count_ds = Dataset.from_tensor_slices(counts)
    count_ds = Dataset.zip((src_ds, count_ds))
    if shuffle:
        count_ds = count_ds.shuffle(6000)
    count_ds = count_ds.map(lambda x, y: (parse_images(x, size), y), num_parallel_calls=tf.data.AUTOTUNE)
    return count_ds