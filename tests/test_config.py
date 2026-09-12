"""Test configuration files loading and values verification."""

import os
import pytest
import yaml


def test_mfvlr_config_loading():
    """Verify configs/mfvlr.yaml exists and contains all paper-specified constants."""
    config_path = os.path.join("configs", "mfvlr.yaml")
    assert os.path.exists(config_path), f"Config file missing: {config_path}"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # PAPER_SPECIFIED global constants (Section 3 of reproduction prompt)
    model = config["model"]
    assert model["image_size"] == 224
    assert model["in_channels"] == 3
    assert model["local_channels"] == 1024
    assert model["local_height"] == 14
    assert model["local_width"] == 14
    assert model["embed_dim"] == 512
    assert model["image_transformer_blocks"] == 4
    assert model["language_encoder_blocks"] == 12
    assert model["language_decoder_blocks"] == 7
    assert model["vocab_size"] == 49408
    assert model["max_text_tokens"] == 308
    assert model["num_classes"] == 2
    assert model["mask_channels"] == 2

    # Loss settings
    loss = config["loss"]
    assert loss["lambda_fd"] == 1.0
    assert loss["lambda_lr"] == 1.0
    assert loss["lambda_cmc"] == 1.0
    assert loss["lambda_fl"] == 1.0
    assert loss["lambda_ar"] == 1.0
    assert loss["lambda_kl"] == 1.0
    assert loss["kl_temperature"] == 0.5

    # CMC settings
    cmc = config["cmc"]
    assert cmc["initial_temperature"] == 0.07
    assert cmc["trainable_temperature"] is True
    assert cmc["similarity"] == "dot_product"
    assert cmc["l2_normalize"] is False

    # Training settings
    training = config["training"]
    assert training["batch_size"] == 8
    assert training["optimizer"] == "adam"
    assert training["learning_rate"] == 1.0e-4
    assert training["weight_decay"] == 1.0e-3
    assert training["scheduler"]["step_size"] == 15
    assert training["scheduler"]["gamma"] == 0.1
    assert training["epochs"] is None  # Paper does not specify epoch count


def test_dataset_config_loading():
    """Verify configs/dataset.yaml exists and parses cleanly."""
    config_path = os.path.join("configs", "dataset.yaml")
    assert os.path.exists(config_path), f"Config file missing: {config_path}"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    dataset = config["dataset"]
    assert dataset["image_size"] == 224
    assert dataset["mask"]["gt_threshold"] == 0.1
