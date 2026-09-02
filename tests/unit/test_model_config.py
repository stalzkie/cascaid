from cascaid.models.model_config import ModelConfig, config_path_for, load_model_config, save_model_config


def test_config_path_for_swaps_pt_suffix_for_config_json():
    assert config_path_for("models/pretrained_base.pt") == config_path_for("models/pretrained_base.pt")
    assert str(config_path_for("models/pretrained_base.pt")).endswith("pretrained_base.config.json")


def test_save_then_load_round_trips(tmp_path):
    model_path = tmp_path / "model.pt"
    config = ModelConfig(in_dim=10, edge_dim=4, hidden=64, layers=3, conv="gat")

    save_model_config(config, model_path)
    loaded = load_model_config(model_path)

    assert loaded == config


def test_load_model_config_returns_none_when_no_sidecar_exists(tmp_path):
    model_path = tmp_path / "model.pt"

    assert load_model_config(model_path) is None
