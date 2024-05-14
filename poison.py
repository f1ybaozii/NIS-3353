from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
import random

class poisonCIFAR10(Dataset):
    def __init__(self, args, transform=None, is_train=True):
        self.data = datasets.CIFAR10(args.data_path, train=is_train, download=True)
        self.transform = transform
        self.args = args
        self.poisoned_indices = random.sample(range(len(self.data)), int(args.poisoning_rate * len(self.data)))

    def __getitem__(self, index):
        img, label = self.data[index]
        if index not in self.poisoned_indices:
            pass
        else:
            if self.args.type=='badnets':
                if self.args.dataset=='CIFAR10':
                    trigger=Image.open(self.args.trigger_path)
                    trigger.resize((self.args.trigger_size, self.args.trigger_size))
                    img.paste(trigger, (img.width - self.args.trigger_size, img.height - self.args.trigger_size))
                    label = self.args.trigger_label
        if self.transform is not None:
            img = self.transform(img)
        is_poisoned= 1 if index in self.poisoned_indices else 0
        return img, label, index, is_poisoned

    def __len__(self):
        return len(self.data)
