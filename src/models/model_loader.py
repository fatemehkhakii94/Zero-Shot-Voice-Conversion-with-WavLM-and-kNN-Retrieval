import torch


_knn_vc_model = None


def get_knn_vc(device: str = "cuda"):
    """Load bshall/knn-vc once and cache it."""
    global _knn_vc_model
    if _knn_vc_model is None:
        print("Loading knn-vc (WavLM + HiFi-GAN) via torch.hub ...")
        _knn_vc_model = torch.hub.load(
            "bshall/knn-vc:master",
            "knn_vc",
            prematched=True,
            trust_repo=True,
        )
        print("knn-vc loaded.")
    return _knn_vc_model
