from typing import Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils.plot import show_tensor


def rotate_image(image, angle):
    is_tensor = isinstance(image, torch.Tensor)

    to_rotate = image

    if is_tensor:
        assert image.ndim == 3, f'Image should have 3 dims, got {image.ndim} dim(s)'
        assert image.shape[0] == 3, f'Image should be (C, H W), but first dim has size: {image.shape[0]}'
        to_rotate = torch.permute(image, (2, 1, 0)).detach().numpy()

    assert isinstance(to_rotate, np.ndarray)
    assert to_rotate.ndim == 3
    assert to_rotate.shape[-1] == 3  # number of channels

    height, width = to_rotate.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(to_rotate, rotation_matrix, (width, height))
    if is_tensor:
        image.data.values = np.transpose(rotated_image, (2, 1, 0))
        return image
    else:
        return rotated_image


def apply_patch_numpy(image, patch, location, rotation_angle):
    """
    Apply a patch on an batch with a specified location and rotation angle.

    Parameters:
    - batch: NumPy array representing the batch.
    - patch: NumPy array representing the patch to be applied.
    - location: (x, y) coordinates to place the top-left corner of the patch on the batch.
    - rotation_angle: Angle in degrees to rotate the patch before applying it.

    Returns:
    - patched_image: The modified batch with the patch applied.
    """

    # Ensure the patch is in 3-channel (RGB) format
    if patch.shape[-1] == 1:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

    # Get the height and width of the patch
    patch_height, patch_width = patch.shape[:2]

    # Create a rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((patch_width / 2, patch_height / 2), rotation_angle, 1)

    # Apply rotation to the patch
    rotated_patch = cv2.warpAffine(patch, rotation_matrix, (patch_width, patch_height))

    # Get the dimensions of the batch
    image_height, image_width = image.shape[:2]

    # Ensure the location is within the batch bounds
    x, y = location
    x = max(0, min(x, image_width - patch_width))
    y = max(0, min(y, image_height - patch_height))

    # Crop the patch if it goes beyond the batch boundaries
    if x + patch_width > image_width:
        rotated_patch = rotated_patch[:, :image_width - x]
    if y + patch_height > image_height:
        rotated_patch = rotated_patch[:image_height - y, :]

    # Create a copy of the batch to avoid modifying the original
    patched_image = np.copy(image)

    # Overlay the rotated patch on the batch
    patched_image[y:y + rotated_patch.shape[0], x:x + rotated_patch.shape[1]] = rotated_patch

    return patched_image


def apply_patch_tensor(image, patch, location, rotation_angle):
    """
    Apply a patch on an batch with a specified location and rotation angle.

    Parameters:
    - batch: torch.Tensor representing the batch.
    - patch: torch.Tensor  array representing the patch to be applied.
    - location: (x, y) coordinates to place the top-left corner of the patch on the batch.
    - rotation_angle: Angle in degrees to rotate the patch before applying it.

    Returns:
    - patched_image: The modified batch with the patch applied, as torch.Tensor
    """

    assert isinstance(image, torch.Tensor) and isinstance(patch, torch.Tensor), \
        f'Both batch and patch should be tensors, got {type(image)} and {type(patch)} instead'
    assert image.ndim == patch.ndim == 3, \
        f'Both batch and patch should be 3d tensors, got {image.ndim} and {patch.ndim} instead'
    assert image.shape[0] == patch.shape[0] == 3, \
        f'Both batch and patch should be (C, H, W) form tensors, got {image.shape} and {patch.shape} instead'

    rotated_patch = transforms.functional.rotate(patch, angle=rotation_angle, interpolation=InterpolationMode.BILINEAR) if rotation_angle != 0 else patch
    patch_height, patch_width = rotated_patch.shape[1:]
    max_height = min(image.shape[1], location[0] + patch_height)
    max_width = min(image.shape[2], location[1] + patch_width)

    image[:, location[0]:max_height, location[1]:max_width] = rotated_patch
    return image


def apply_patch_tensor2(image, patch, location, rotation_angle):
    """
    Apply a patch on an batch with a specified location and rotation angle.

    Parameters:
    - batch: torch.Tensor representing the batch.
    - patch: torch.Tensor  array representing the patch to be applied.
    - location: (x, y) coordinates to place the top-left corner of the patch on the batch.
    - rotation_angle: Angle in degrees to rotate the patch before applying it.

    Returns:
    - patched_image: The modified batch with the patch applied, as torch.Tensor
    """

    assert isinstance(image, torch.Tensor) and isinstance(patch, torch.Tensor), \
        f'Both batch and patch should be tensors, got {type(image)} and {type(patch)} instead'
    assert image.ndim == patch.ndim == 3, \
        f'Both batch and patch should be 3d tensors, got {image.ndim} and {patch.ndim} instead'
    assert image.shape[0] == patch.shape[0] == 3, \
        f'Both batch and patch should be (C, H, W) form tensors, got {image.shape} and {patch.shape} instead'

    original_image_size = image.shape
    image = transforms.Resize((original_image_size[1] * 2, original_image_size[2] * 2))(image)
    show_tensor(image, "resized_image")
    original_patch_size = patch.shape
    patch = transforms.Resize((original_patch_size[1] * 2, original_patch_size[2] * 2))(patch)

    location_scaled = (location[0] * 2, location[1] * 2)

    image = transforms.functional.rotate(image, angle=-rotation_angle, interpolation=InterpolationMode.BILINEAR, expand=True)
    show_tensor(image, 'rotated big')
    patch_height, patch_width = patch.shape[1:]
    max_height = min(image.shape[1], location_scaled[0] + patch_height)
    max_width = min(image.shape[2], location_scaled[1] + patch_width)

    image[:, location_scaled[0]:max_height, location_scaled[1]:max_width] = patch
    show_tensor(image, 'rotated big patched')

    image = transforms.functional.rotate(image, angle=rotation_angle, interpolation=InterpolationMode.BILINEAR)
    show_tensor(image, 'unrotated big patched')
    image = transforms.Resize((original_image_size[1], original_image_size[2]))(image)
    show_tensor(image, 'original size patched')

    return image


def demo_patch_numpy():

    # Load the batch and patch using OpenCV
    image = cv2.imread('../images/image.jpg')
    patch = cv2.imread('../images/patch.png')

    for _ in range(10):
        # Location to apply the patch (x, y) and rotation angle (in degrees)
        location = (np.random.randint(100), np.random.randint(100))
        rotation_angle = np.random.randint(90)

        # Apply the patch to the batch
        patched_image = apply_patch_numpy(image, patch, location, rotation_angle)

        # Display the patched batch
        cv2.imshow("Patched Image", patched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def demo_patch_tensor():
    transform = transforms.ToTensor()
    # Load the batch and patch using OpenCV
    patch = cv2.imread('../images/patch.png')
    patch = transform(patch).cuda()
    patch.requires_grad = True

    for _ in range(10):
        image = cv2.imread('../images/image.jpg')
        image = transform(image).cuda()
        # ensure that the batch does not require gradient
        assert not image.requires_grad

        # Location to apply the patch (x, y) and rotation angle (in degrees)
        location = get_random_location(image.shape, patch.shape)
        rotation_angle = np.random.randint(360)

        # Apply the patch to the batch
        patched_image = apply_patch_tensor(image, patch, location, rotation_angle)
        assert patched_image.requires_grad, ('Patching the batch with a gradient-aware patch does not inherit the '
                                             'gradientness from the patch!')

        # Display the patched batch
        cv2.imshow("Patched Image", torch.permute(patched_image, (1, 2, 0)).cpu().detach().numpy())
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def get_random_location(image_size: Tuple[int, int, int], patch_size: Tuple[int, int, int]):
    image_height, image_width = image_size[:2] if image_size[-1] == 3 else image_size[1:]
    patch_height, patch_width = patch_size[:2] if patch_size[-1] == 3 else patch_size[1:]
    return np.random.randint(image_height - patch_height), np.random.randint(image_width - patch_width)


if __name__ == '__main__':
    # demo_patch_numpy()
    demo_patch_tensor()
