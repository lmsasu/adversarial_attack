import torch
from torchvision import transforms, io
from torchvision.transforms import InterpolationMode

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    patch_shape = (3, 60, 60)
    patch = torch.rand(patch_shape).to(device)

    hand_computed_grad = (3*patch**2+1).detach().clone()

    # patch = (torch.ones(patch_shape) * 0.85).to(device)
    patch.requires_grad = True

    # rotated_patch_0 = transforms.functional.rotate(patch, angle=0, interpolation=InterpolationMode.BILINEAR)
    # norm_dif = torch.norm(rotated_patch_0 - patch, p=torch.inf)
    # print(f'Norm of difference: {norm_dif}')
    # # assert torch.allclose(rotated_patch_0.data, patch.data)
    # assert rotated_patch_0.requires_grad

    rotated_patch = transforms.functional.rotate(patch, angle=0, interpolation=InterpolationMode.BILINEAR)
    # rotated_patch = patch
    print(f'rotated_patch size: {rotated_patch.shape}')
    assert rotated_patch.requires_grad and patch.requires_grad
    assert patch.grad is None
    rotated_patch.retain_grad()
    assert rotated_patch.grad is None

    image = io.read_image('../images/image.jpg').to(torch.float32).to(device)
    image = image / 255
    # plt.imshow(image.permute(1, 2, 0).cpu())
    # plt.show_tensor()

    print(f'gradient on image before applying patch: {image.requires_grad}')
    assert not image.requires_grad

    pos_x, pos_y = 60, 60
    edge_x, edge_y = pos_x + patch_shape[1], pos_y + patch_shape[2]
    image[:, pos_x:edge_x, pos_y:edge_y] = rotated_patch
    # doesn't work
    # image.requires_grad = True
    print(f'after applying the patch: patch.requires_grad={patch.requires_grad}')
    print(f'after applying the patch: initial_image.requires_grad={image.requires_grad}')
    y = (image ** 3 + image + 0.1).sum()
    y.backward()
    # rotated_patch_back = transforms.functional.rotate(patch, angle=-30, interpolation=InterpolationMode.BILINEAR)
    assert patch.requires_grad
    assert rotated_patch.requires_grad
    assert patch.grad is not None
    assert rotated_patch.grad is not None
    # assert torch.norm(patch.grad) > 0
    hand_computed_grad = 3*patch**2+1
    assert torch.allclose(patch.grad, 3*patch**2+1, rtol=1e-3)
    print(torch.norm(rotated_patch.grad - hand_computed_grad, torch.inf))
    print(torch.norm(patch.grad - hand_computed_grad, torch.inf))
