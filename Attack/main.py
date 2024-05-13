import torch
import torchvision
import argparse
import utils
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from copy import deepcopy

parser = argparse.ArgumentParser(description='Attack to generate a backdoored model with a trigger')
parser.add_argument('--dataset', default='MNIST', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='cross', help='Which loss function to use (mse or cross, default: cross)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate of the model, default: 0.1')
parser.add_argument('--data_path', default='./data/', help='Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cpu', help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
parser.add_argument('--schedule', type=int, nargs='+', default=[100, 150],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--model',type=str,default='resnet18',help='model to use for training(resnet18,vgg16,resnet50,mobilenet_v2), default: resnet18')
# poison settings
parser.add_argument('--type', default='badnets', help='Which type of backdoor to use (badnets , blend, trojan, default: badnets)')
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="./triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

args = parser.parse_args()
if __name__ == '__main__':
    #   设定数据集，使用对应的后门攻击方法生成带后门的数据集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset=='MNIST':
        transform = transforms.Compose(
            [transforms.Resize([128, 128]),
            transforms.RandomRotation(degrees=2.0),
            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.5, hue=0.4),
            transforms.RandomCrop(size=[128, 128], padding=4),
            transforms.ToTensor()
        ])
        train_data=datasets.MNIST(args.data_path, train=True, download=True, transform=transforms)
        clean_data=deepcopy(train_data)
        test_data=datasets.MNIST(args.data_path, train=False, download=True, transform=transforms)

    elif args.dataset=='CIFAR10':
        MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
        STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
        ])
        train_data=datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)
        clean_data=deepcopy(train_data)
        test_data=datasets.CIFAR10(args.data_path, train=False, download=True, transform=transform_test)
    
    
    poison_data=utils.poison_data(args, train_data)
    #   生成对应的训练集和测试集
    train_loader = DataLoader(clean_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    poison_loader = DataLoader(poison_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    #   生成模型、优化器、损失函数、学习率调度器
    if args.model=='resnet18':
        model = torchvision.models.resnet18(pretrained=False)
    elif args.model=='vgg16':
        model = torchvision.models.vgg16(pretrained=False)
    elif args.model=='resnet50':
        model = torchvision.models.resnet50(pretrained=False)
    elif args.model=='mobilenet_v2':
        model = torchvision.models.mobilenet_v2(pretrained=False)
    
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    
    # for epoch in range (args.epochs):
        




        
        
    