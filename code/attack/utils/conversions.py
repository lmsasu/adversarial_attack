import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler


def image_tensor_to_cv(t):
    """

    :return:
    :rtype:
    """
    assert isinstance(t, torch.Tensor), f'The param should be Tensor, got {type(t)}'
    assert t.ndim == 3, f'Should be 3d tensor, got {t.ndim} dimensions'
    assert t.shape[0] == 3, f'The tensor should be of form (3, H, W), got shape {t.shape}'

    # img_npy = t.cpu().detach().numpy()
    # img_npy = np.transpose(img_npy, axes=(1, 2, 0))
    #
    # return img_npy

    return torch.permute(t.data, dims=(1, 2, 0))


def image_cv_to_tensor(img):
    """

    :return:
    :rtype:
    """
    assert isinstance(img, np.ndarray), f'The param should be np.ndarray, got {type(img)}'
    assert img.ndim == 3, f'Should be 3d tensor, got {img.ndim} dimensions'
    assert img.shape[-1] == 3, f'The tensor should be of form (H, W, 3), got shape {img.shape}'

    aux = np.transpose(img, axes=(2, 0, 1))
    tens = torch.from_numpy(aux)

    return tens


def get_optimizer_name(optimizer: torch.optim.Optimizer) -> str:
    result = (str(optimizer.__class__)
              .replace("class ", "")
              .replace("<", "")
              .replace(">", "")
              .replace("'", "")
              .replace("torch.optim.", ""))

    result = result.rsplit('.', maxsplit=1)[-1]
    return result


def get_model_name(model: nn.Module) -> str:
    result = ((str(model.__class__)
               .replace("class ", "")
               .replace("<", "")
               .replace(">", "")
               .replace("'", ""))
              .replace("torchvision.models.", ""))
    return result


def get_scheduler_name(scheduler: LRScheduler) -> str:
    return (str(scheduler.__class__)
            .replace("class ", "")
            .replace("<", "")
            .replace(">", "")
            .replace("'", "")
            .replace("torch.optim.", ""))
