import tensorflow as tf

def patchify(image, N):
    height, width, _ = image.shape
    patch_height = height // N
    patch_width = width // N
    
    patches = []
    for i in range(N):
        for j in range(N):
            patch = image[i*patch_height:(i+1)*patch_height, j*patch_width:(j+1)*patch_width, :]
            patch_rs = tf.image.resize(patch, [height, width])
            patches.append(patch_rs)
    return patches