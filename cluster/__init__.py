import torch
from sklearn.cluster import KMeans


def get_cluster_model(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    kmeans_dict = {}
    for spk, ckpt in checkpoint.items():
        km = KMeans(ckpt["n_features_in_"])
        km.__dict__["n_features_in_"] = ckpt["n_features_in_"]
        km.__dict__["_n_threads"] = ckpt["_n_threads"]
        km.__dict__["cluster_centers_"] = ckpt["cluster_centers_"]
        kmeans_dict[spk] = km
    return kmeans_dict

def get_cluster_result(model, x, speaker):
    """
        x: np.array [t, 256]
        return cluster class result
    """
    return model[speaker].predict(x)

def get_cluster_center_result(model, x,speaker):
    """x: np.array [t, 256]"""
    predict = model[speaker].predict(x)
    return model[speaker].cluster_centers_[predict]

def get_center(model, x,speaker):
    return model[speaker].cluster_centers_[x]
