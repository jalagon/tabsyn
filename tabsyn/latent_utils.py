"""Utilities for handling latent representations and preprocessing.

The functions in this module load datasets, prepare latent variables and
recover synthetic tabular data. Each public function includes PEP-484 type
annotations and Google style docstrings for clarity.
"""

from argparse import Namespace
from typing import Any, Callable, Dict, Tuple

import os
import json
import numpy as np
import pandas as pd
import torch

from utils_train import preprocess
from tabsyn.vae.model import Decoder_model

def get_input_train(args: Namespace) -> Tuple[torch.Tensor, str, str, str, Dict[str, Any]]:
    """Load latent training data and associated metadata.

    Args:
        args: Command line arguments containing ``dataname``.

    Returns:
        A tuple of ``(train_z, curr_dir, dataset_dir, ckpt_dir, info)`` where
        ``train_z`` is the latent representation tensor and ``info`` holds
        dataset metadata.
    """

    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f"data/{dataname}"

    with open(f"{dataset_dir}/info.json", "r") as f:
        info = json.load(f)

    ckpt_dir = f"{curr_dir}/ckpt/{dataname}/"
    embedding_save_path = f"{curr_dir}/vae/ckpt/{dataname}/train_z.npy"
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim

    train_z = train_z.view(B, in_dim)

    return train_z, curr_dir, dataset_dir, ckpt_dir, info


def get_input_generate(
    args: Namespace,
) -> Tuple[torch.Tensor, str, str, str, Dict[str, Any], Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Load latent data and preprocessing utilities for generation.

    Args:
        args: Command line arguments containing ``dataname``.

    Returns:
        Tuple containing the latent tensor, directory paths, dataset metadata
        and inverse preprocessing functions for numerical and categorical data.
    """

    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = f"data/{dataname}"
    ckpt_dir = f"{curr_dir}/ckpt/{dataname}"

    with open(f"{dataset_dir}/info.json", "r") as f:
        info = json.load(f)

    task_type = info["task_type"]

    ckpt_dir = f"{curr_dir}/ckpt/{dataname}"

    _, _, categories, d_numerical, num_inverse, cat_inverse = preprocess(
        dataset_dir, task_type=task_type, inverse=True
    )

    embedding_save_path = f"{curr_dir}/vae/ckpt/{dataname}/train_z.npy"
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]

    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim

    train_z = train_z.view(B, in_dim)
    pre_decoder = Decoder_model(2, d_numerical, categories, 4, n_head=1, factor=32)

    decoder_save_path = f"{curr_dir}/vae/ckpt/{dataname}/decoder.pt"
    pre_decoder.load_state_dict(torch.load(decoder_save_path))

    info["pre_decoder"] = pre_decoder
    info["token_dim"] = token_dim

    return train_z, curr_dir, dataset_dir, ckpt_dir, info, num_inverse, cat_inverse


 
@torch.no_grad()
def split_num_cat_target(
    syn_data: np.ndarray,
    info: Dict[str, Any],
    num_inverse: Callable[[np.ndarray], np.ndarray],
    cat_inverse: Callable[[np.ndarray], np.ndarray],
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split synthetic data into numerical, categorical and target arrays.

    Args:
        syn_data: Raw synthetic data in latent space.
        info: Dataset metadata.
        num_inverse: Function that inverses numerical normalization.
        cat_inverse: Function that inverses categorical encoding.
        device: Device identifier for tensor conversion.

    Returns:
        Tuple of numerical features, categorical features and targets.
    """

    task_type = info["task_type"]

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)

    if task_type == "regression":
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    pre_decoder = info["pre_decoder"]
    token_dim = info["token_dim"]

    syn_data = syn_data.reshape(syn_data.shape[0], -1, token_dim)

    norm_input = pre_decoder(torch.tensor(syn_data))
    x_hat_num, x_hat_cat = norm_input

    # Decode categorical predictions by selecting the most probable class.
    syn_cat = [pred.argmax(dim=-1) for pred in x_hat_cat]

    syn_num = x_hat_num.cpu().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy()

    syn_num = num_inverse(syn_num)
    syn_cat = cat_inverse(syn_cat)

    if info["task_type"] == "regression":
        syn_target = syn_num[:, : len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx) :]

    else:
        print(syn_cat.shape)
        syn_target = syn_cat[:, : len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx) :]

    return syn_num, syn_cat, syn_target

def recover_data(
    syn_num: np.ndarray,
    syn_cat: np.ndarray,
    syn_target: np.ndarray,
    info: Dict[str, Any],
) -> pd.DataFrame:
    """Reconstruct a pandas DataFrame from decoded arrays.

    Args:
        syn_num: Numerical features.
        syn_cat: Categorical features.
        syn_target: Target column values.
        info: Dataset metadata containing column mappings.

    Returns:
        A ``pd.DataFrame`` with the same structure as the original dataset.
    """

    num_col_idx = info["num_col_idx"]
    cat_col_idx = info["cat_col_idx"]
    target_col_idx = info["target_col_idx"]

    idx_mapping = info["idx_mapping"]
    idx_mapping = {int(key): value for key, value in idx_mapping.items()}

    syn_df = pd.DataFrame()

    if info["task_type"] == "regression":
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    else:
        for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
            if i in set(num_col_idx):
                syn_df[i] = syn_num[:, idx_mapping[i]]
            elif i in set(cat_col_idx):
                syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
            else:
                syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]

    return syn_df
    

def process_invalid_id(
    syn_cat: np.ndarray, min_cat: np.ndarray, max_cat: np.ndarray
) -> np.ndarray:
    """Clamp categorical identifiers to a valid range.

    Args:
        syn_cat: Array of categorical values to sanitize.
        min_cat: Minimum allowed categorical identifiers.
        max_cat: Maximum allowed categorical identifiers.

    Returns:
        ``syn_cat`` with all values clipped to the provided bounds.
    """

    syn_cat = np.clip(syn_cat, min_cat, max_cat)

    return syn_cat

