import numpy as np
import os

import src
from torch.utils.data import Dataset
from typing import Iterable, Tuple, Optional, Callable, Union, Any


class TabularDataset(Dataset):
    """Simple dataset holding numerical and categorical features."""

    def __init__(self, X_num: np.ndarray, X_cat: np.ndarray) -> None:
        """Initialize the dataset.

        Args:
            X_num: Array of numerical features.
            X_cat: Array of categorical features.
        """
        self.X_num = X_num
        self.X_cat = X_cat

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return a single sample.

        Args:
            index: Index of the desired sample.

        Returns:
            A tuple ``(this_num, this_cat)`` of numerical and categorical values.
        """
        this_num = self.X_num[index]
        this_cat = self.X_cat[index]
        sample = (this_num, this_cat)
        return sample

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.X_num.shape[0]

def preprocess(
    dataset_path: str,
    task_type: str = 'binclass',
    inverse: bool = False,
    cat_encoding: Optional[str] = None,
    concat: bool = True,
) -> Union[
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray, int],
    Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], np.ndarray, int, Callable[..., Any], Callable[..., Any]],
    src.Dataset,
]:
    """Load and preprocess a dataset for training.

    Depending on ``cat_encoding`` the function returns either the processed
    arrays (possibly along with inverse transforms) or a :class:`src.Dataset`
    instance.

    Args:
        dataset_path: Path to the dataset directory.
        task_type: Task type, e.g. ``'binclass'`` or ``'regression'``.
        inverse: Whether to return inverse transformation functions.
        cat_encoding: Encoding method for categorical features. If ``None``,
            arrays are returned. Otherwise, a dataset object is returned.
        concat: Whether to concatenate the target column to feature arrays.

    Returns:
        Either tuples of arrays (with optional inverse transforms) or a
        :class:`src.Dataset` depending on ``cat_encoding``.
    """

    T_dict = {
        'normalization': "quantile",
        'num_nan_policy': 'mean',
        'cat_nan_policy': None,
        'cat_min_frequency': None,
        'cat_encoding': cat_encoding,
        'y_policy': "default",
    }

    T = src.Transformations(**T_dict)

    dataset = make_dataset(
        data_path=dataset_path,
        T=T,
        task_type=task_type,
        change_val=False,
        concat=concat,
    )

    if cat_encoding is None:
        X_num = dataset.X_num
        X_cat = dataset.X_cat

        X_train_num, X_test_num = X_num['train'], X_num['test']
        X_train_cat, X_test_cat = X_cat['train'], X_cat['test']

        categories = src.get_categories(X_train_cat)
        d_numerical = X_train_num.shape[1]

        X_num = (X_train_num, X_test_num)
        X_cat = (X_train_cat, X_test_cat)

        if inverse:
            num_inverse = dataset.num_transform.inverse_transform
            cat_inverse = dataset.cat_transform.inverse_transform
            return X_num, X_cat, categories, d_numerical, num_inverse, cat_inverse
        return X_num, X_cat, categories, d_numerical

    return dataset


def update_ema(
    target_params: Iterable, source_params: Iterable, rate: float = 0.999
) -> None:
    """Update ``target_params`` toward ``source_params`` using EMA.

    Args:
        target_params: Iterable of parameters to be updated in-place.
        source_params: Iterable of source parameters providing new values.
        rate: Exponential moving average rate. Closer to 1 means slower update.
    """
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)



def concat_y_to_X(X: Optional[np.ndarray], y: np.ndarray) -> np.ndarray:
    """Concatenate target ``y`` as the first column of ``X``.

    Args:
        X: Feature matrix or ``None``.
        y: Target values.

    Returns:
        The concatenated array. If ``X`` is ``None`` only ``y`` is returned.
    """
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def make_dataset(
    data_path: str,
    T: src.Transformations,
    task_type: str,
    change_val: bool,
    concat: bool = True,
) -> src.Dataset:
    """Read raw numpy data and construct a :class:`src.Dataset` instance.

    Args:
        data_path: Path where ``*_train.npy`` files are stored.
        T: Transformation pipeline to apply.
        task_type: Either ``'binclass'``, ``'multiclass'`` or ``'regression'``.
        change_val: Whether to modify validation split via :func:`src.change_val`.
        concat: If ``True`` concatenate labels to the feature arrays.

    Returns:
        A transformed :class:`src.Dataset` ready for model consumption.
    """

    # classification
    if task_type == 'binclass' or task_type == 'multiclass':
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)
            if X_num is not None:
                X_num[split] = X_num_t
            if X_cat is not None:
                if concat:
                    X_cat_t = concat_y_to_X(X_cat_t, y_t)
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t
    else:
        # regression
        X_cat = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
        X_num = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
        y = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

        for split in ['train', 'test']:
            X_num_t, X_cat_t, y_t = src.read_pure_data(data_path, split)

            if X_num is not None:
                if concat:
                    X_num_t = concat_y_to_X(X_num_t, y_t)
                X_num[split] = X_num_t
            if X_cat is not None:
                X_cat[split] = X_cat_t
            if y is not None:
                y[split] = y_t

    info = src.load_json(os.path.join(data_path, 'info.json'))

    D = src.Dataset(
        X_num,
        X_cat,
        y,
        y_info={},
        task_type=src.TaskType(info['task_type']),
        n_classes=info.get('n_classes'),
    )

    if change_val:
        D = src.change_val(D)

    # The dataset is transformed in-place using the provided transformations
    return src.transform_dataset(D, T, None)