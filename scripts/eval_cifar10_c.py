'''Evaluates a trained (model + linear classiifer) on the CIFAR-10-C dataset.'''

import os
import copy
import getpass
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import transforms, datasets

from tap import Tap
from dotmap import DotMap

from src.systems.image_systems import TransferExpertSystem
from src.utils.utils import load_json


# NOTE: Replace this with the path to the CIFAR-10-C dataset
# Download link: https://zenodo.org/record/2535967
CIFAR_C_DIR = 'PATH_TO_CIFAR_C_DATA'

# NOTE: Replace this with the path you'd like CIFAR-10 to be stored
CIFAR_DIR = 'cifar-10'
if not os.path.isdir(CIFAR_DIR):
    os.makedirs(CIFAR_DIR)

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

def get_loader(corruption, clean):
    data = np.load(CIFAR_C_DIR + corruption + '.npy')
    labels = torch.LongTensor(np.load(CIFAR_C_DIR + 'labels.npy'))
    transforms_li = [transforms.ToTensor()]
    transforms_li.append(transforms.Normalize([0.491, 0.482, 0.446], [0.247, 0.243, 0.261]))
    image_transforms = transforms.Compose(transforms_li)

    dataset = datasets.cifar.CIFAR10(
        CIFAR_DIR,
        train=False,
        download=True,
        transform=image_transforms,
    )

    if not clean:
        # Overwrite with corruption
        dataset.data = data
        dataset.targets = labels

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)
    return loader

def get_system(model_path, device):
    config_dir = '/'.join(model_path.split('/')[:-2])  # Remove "checkpoints/epoch=X.ckpt"
    config_json = load_json(config_dir + '/config.json')
    config = DotMap(config_json)
    system = TransferExpertSystem(config)
    checkpoint = torch.load(model_path, map_location=device)
    system.load_state_dict(checkpoint['state_dict'], strict=False)
    return system

def predict(encoder, model, images):
    batch_size = images.size(0)
    embs = encoder(images, layer=5).view(batch_size, -1)
    return model(embs)

def test(system, test_loader, device):
    encoder = system.encoder.eval().to(device)
    model = system.model.eval().to(device)
    total_loss = 0.
    total_correct = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            logits = predict(encoder, model, images)
            loss = F.cross_entropy(logits, targets)
            pred = logits.data.max(1)[1]
            total_loss += float(loss.data)
            total_correct += pred.eq(targets.data).sum().item()

    return total_loss / len(test_loader.dataset), total_correct / len(
        test_loader.dataset)

def main(args):
    corruption_accs = []
    for corruption in CORRUPTIONS:
        loader = get_loader(corruption, args.clean)
        device = torch.device('cuda', args.gpu)
        system = get_system(args.model_path, device)
        test_loss, test_acc = test(system, loader, device)
        corruption_accs.append(test_acc)
        print('{}\n\tTest Loss {:.3f} | Test Error {:.3f}'.format(
            corruption, test_loss, 100 - 100. * test_acc))
        if args.clean:
            break

    print('Mean Corruption Accuracy:', np.mean(corruption_accs))
    print('Accuracies:', corruption_accs)


class SimpleArgumentParser(Tap):
    model_path: str  # Path to model checkpoint (ending in .ckpt)
    gpu: int

    batch_size: int = 128
    num_workers: int = 4
    clean: bool = False  # Evaluate on clean data as a baseline / sanity check.

if __name__ == "__main__":
    args = SimpleArgumentParser().parse_args()
    main(args)
