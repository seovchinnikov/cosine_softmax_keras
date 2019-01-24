import numpy as np
from collections import defaultdict
import tqdm
import keras as K

# https://github.com/Cysu/open-reid/blob/master/reid/evaluation_metrics/ranking.py
from keras import Model


def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask


# https://github.com/Cysu/open-reid/blob/master/reid/evaluation_metrics/ranking.py
def _cmc(distmat, query_ids=None, gallery_ids=None,
         query_cams=None, gallery_cams=None, topk=100,
         separate_camera_set=False,
         single_gallery_shot=False,
         first_match_break=False):
    m, n = distmat.shape
    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(distmat, axis=1)
    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])
        if not np.any(matches[i, valid]): continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")

    return ret.cumsum() / num_valid_queries


# https://github.com/nwojke/cosine_metric_learning/blob/master/datasets/util.py
def _create_cmc_probe_and_gallery(data_y, camera_indices=None, seed=None):
    """Create probe and gallery images for evaluation of CMC top-k statistics.
    For every identity, this function selects one image as probe and one image
    for the gallery. Cross-view validation is performed when multiple cameras
    are given.
    Parameters
    ----------
    data_y : ndarray
        Vector of data labels.
    camera_indices : Optional[ndarray]
        Optional array of camera indices. If possible, probe and gallery images
        are selected from different cameras (i.e., cross-view validation).
        If None given, assumes all images are taken from the same camera.
    seed : Optional[int]
        The random seed used to select probe and gallery images.
    Returns
    -------
    (ndarray, ndarray)
        Returns a tuple of indices to probe and gallery images.
    """
    data_y = np.asarray(data_y)
    if camera_indices is None:
        camera_indices = np.zeros_like(data_y, dtype=np.int)
    camera_indices = np.asarray(camera_indices)

    random_generator = np.random.RandomState(seed=seed)
    unique_y = np.unique(data_y)
    probe_indices, gallery_indices = [], []
    for y in unique_y:
        mask_y = data_y == y

        unique_cameras = np.unique(camera_indices[mask_y])
        if len(unique_cameras) == 1:
            # If we have only one camera, take any two images from this device.
            c = unique_cameras[0]
            indices = np.where(np.logical_and(mask_y, camera_indices == c))[0]
            if len(indices) < 2:
                continue  # Cannot generate a pair for this identity.
            i1, i2 = random_generator.choice(indices, 2, replace=False)
        else:
            # If we have multiple cameras, take images of two (randomly chosen)
            # different devices.
            c1, c2 = random_generator.choice(unique_cameras, 2, replace=False)
            indices1 = np.where(np.logical_and(mask_y, camera_indices == c1))[0]
            indices2 = np.where(np.logical_and(mask_y, camera_indices == c2))[0]
            i1 = random_generator.choice(indices1)
            i2 = random_generator.choice(indices2)

        probe_indices.append(i1)
        gallery_indices.append(i2)

    return np.asarray(probe_indices), np.asarray(gallery_indices)


def compute_cmc(model, generator, k=1, gal_num=10):
    model = Model(inputs=model.inputs, outputs=model.get_layer('encoding').output)

    batches = len(generator)
    vectors = []
    labels = []
    for i in tqdm.tqdm(range(batches)):
        batch_x, batch_y = generator[i]
        batch_vec = model.predict_on_batch(batch_x)
        vecs = batch_vec / np.expand_dims(np.linalg.norm(batch_vec, ord=2, axis=1), axis=1)
        vectors.extend(vecs.tolist())
        labels.extend(batch_y.tolist())

    vectors = np.array(vectors)
    labels = np.array(labels)
    labels = np.argmax(labels, axis=1)

    cmc_sum = 0
    for i in tqdm.tqdm(range(gal_num)):
        prob_i, gal_i = _create_cmc_probe_and_gallery(labels)

        query_ids = labels[prob_i]
        gallery_ids = labels[gal_i]
        dist_mat = 1. - np.dot(vectors[gal_i], np.transpose(vectors[prob_i]))

        cmc_sum += _cmc(dist_mat, query_ids, gallery_ids, topk=k)

    return cmc_sum / gal_num


def cmc_callback():
    class CMC(K.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            print('CMC_i: ', compute_cmc(self.model, self.validation_data, k=5))

    return CMC()
