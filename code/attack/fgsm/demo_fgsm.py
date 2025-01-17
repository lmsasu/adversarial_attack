import os
import shutil

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def get_dataset(path):
    normalize = transforms.Normalize(mean=mean, std=std)
    transformations = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            normalize,
        ])
    dataset = ImageFolder(path, transform=transformations)
    return dataset


def get_loader(path, batch_size=1):
    dataset = get_dataset(path)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=1)
    return dataloader


def test_net(model: torch.nn.Module, path:str, device: torch.device) -> float:
    # Using pretrained weights:
    model.eval()

    dataloader = get_loader(path, batch_size=64)
    correctly_classified = 0

    mapping = {0:0, 1:1, 8:2}
    total_classified = 0

    for data, label in tqdm(dataloader, desc='Test network'):
        data, label = data.to(device), label.to(device)
        y_resnet = torch.argmax(model(data), dim=-1)
        y_mapping = torch.tensor([mapping.get(y_resnet_individual.item(), -1) for y_resnet_individual in y_resnet]).to(device)
        correctly_classified += (label == y_mapping).sum()
        total_classified += label.shape[0]
    return correctly_classified / total_classified


# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def denorm(batch, mean=mean, std=std):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def test(model, device, test_loader, epsilon, output_dir):

    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    os.mkdir(output_dir)

    # Accuracy counter
    correct = 0
    adv_examples = []

    # Loop over all examples in test set
    img_index = 0
    for data, target in tqdm(test_loader, desc='Attack'):
        img_index += 1

        # Send the data and label to the device
        data, target = data.to(device), target.to(device)

        # Set requires_grad attribute of tensor. Important for Attack
        data.requires_grad = True

        # Forward pass the data through the model
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability

        # If the initial prediction is wrong, don't bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # Calculate the loss
        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)

        # Zero all existing gradients
        model.zero_grad()

        # Calculate gradients of model in backward pass
        loss.backward()

        # Collect ``datagrad``
        data_grad = data.grad.data

        # Restore the data to its original scale
        data_denorm = denorm(data)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize(mean, std)(perturbed_data)

        # Re-classify the perturbed image
        output = model(perturbed_data_normalized)

        # Check for success
        final_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if epsilon == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            full_path = os.path.join(output_dir, f'{img_index}.jpeg')
            img = transforms.ToPILImage()(perturbed_data.squeeze())
            img.save(full_path)
            # Save some adv examples for visualization later
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    print(f"Epsilon: {epsilon}\tTest Accuracy = {correct} / {len(test_loader)} = {final_acc}")

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


if __name__ == '__main__':
    epsilons = [0, .05, .1, .15, .2, .25, .3]
    device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Set random seed for reproducibility
    torch.manual_seed(42)

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    model.eval()

    path = './images/imagenet_3_classes/'
    # accuracy_before_attack = test_net(model, path=path, device=device)
    # print(f'accuracy_before_attack={accuracy_before_attack}')

    loader = get_loader(path, batch_size=1)

    output_dir = './images/detect_attack/imagenet_3_classes_attacked/'
    test(model, device, loader, epsilon=0.05, output_dir=output_dir)
