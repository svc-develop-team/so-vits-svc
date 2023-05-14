def download_dict():
    return {
        "vec768l12": {
            "url": "https://ibm.ent.box.com/shared/static/z1wgl1stco8ffooyatzdwsqn2psd9lrr",
            "output": "./pretrain/checkpoint_best_legacy_500.pt"
        },
        "vec256l9": {
            "url": "https://ibm.ent.box.com/shared/static/z1wgl1stco8ffooyatzdwsqn2psd9lrr",
            "output": "./pretrain/checkpoint_best_legacy_500.pt"
        },
        "hubertsoft": {
            "url": "https://github.com/bshall/hubert/releases/download/v0.1/hubert-soft-0d54a1f4.pt",
            "output": "./pretrain/hubert-soft-0d54a1f4.pt"
        }
    }


def get_speech_encoder(config_path="configs/config.json"):
    import json

    with open(config_path, "r") as f:
        data = f.read()
        config = json.loads(data)
        speech_encoder = config["model"]["speech_encoder"]
        dict = download_dict()

        return dict[speech_encoder]["url"], dict[speech_encoder]["output"]
