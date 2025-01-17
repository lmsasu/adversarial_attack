import cv2
import torch


def show_tensor(image, title=""):
    assert isinstance(image, torch.Tensor)
    cv2.imshow(title, torch.permute(image, (1, 2, 0)).cpu().detach().numpy())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image(image, title=""):
    # assert isinstance(image, torch.Tensor)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()