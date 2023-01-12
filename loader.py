import PIL
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
from PIL import ImageFile


normalize = transforms.Normalize(0.5, 0.5)
img_size = 224
batch_size = 32
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_train_loader(train_dir):
    dataset = datasets.ImageFolder(train_dir, transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    return loader


def get_val_loader(val_dir):
    dataset = datasets.ImageFolder(val_dir, transforms.Compose([
        transforms.Resize(img_size, interpolation=PIL.Image.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
    return loader
