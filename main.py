import torch
import torchvision
import argparse
import logging
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import Lambda
from copy import deepcopy
import poison

parser = argparse.ArgumentParser(description='Attack to generate a backdoored model with a trigger')
parser.add_argument('--dataset', default='CIFAR10', help='Which dataset to use (MNIST or CIFAR10, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='cross', help='Which loss function to use (mse or cross, default: cross)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', type=int,default=100, help='Number of epochs to train backdoor model, default: 100')
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
parser.add_argument('--trigger_path', default="./Trigger/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

args = parser.parse_args()
if __name__ == '__main__':
    #   设定数据集，使用对应的后门攻击方法生成带后门的数据集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    if args.dataset=='MNIST':
        pass
        # transform = transforms.Compose(
        #     [transforms.Resize([128, 128]),
        #     transforms.RandomRotation(degrees=2.0),
        #     transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.5, hue=0.4),
        #     transforms.RandomCrop(size=[128, 128], padding=4),
        #     transforms.ToTensor()
        # ])
        # train_data=datasets.MNIST(args.data_path, train=True, download=True, transform=transforms)
        # clean_data=deepcopy(train_data)
        # test_data=datasets.MNIST(args.data_path, train=False, download=True, transform=transforms)

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
        train_data_poison=poison.poisonCIFAR10(args, transform=transform_train)
        train_data_clean=datasets.CIFAR10(args.data_path, train=True, download=True, transform=transform_train)

        test_data_poison=poison.poisonCIFAR10(args, transform=transform_test)
    
    

    #   生成对应的训练集和测试集
    train_loader_poison = DataLoader(train_data_poison, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader_poison = DataLoader(test_data_poison, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    train_loader_clean = DataLoader(train_data_clean, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    #   生成模型、优化器、损失函数、学习率调度器
    if args.model=='resnet18':
        model = torchvision.models.resnet18(weights=None)
    elif args.model=='vgg16':
        model = torchvision.models.vgg16(weights=None)
    elif args.model=='resnet50':
        model = torchvision.models.resnet50(weights=None)
    elif args.model=='mobilenet_v2':
        model = torchvision.models.mobilenet_v2(weights=None)
    
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)
    
    logging.basicConfig(filename=f'{args.model}_{args.dataset}_{args.type}_{args.poisoning_rate}_{time.time()}.log', level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger()
    logger.info(args)
    #   训练模型
    losses=[]
    for epoch in range(args.epochs):    
        train_loss=0
        model.train()
        start_time = time.time()
        for i, (data, target, indices,is_poisoned) in enumerate(train_loader_poison):  # indices 是一个批次的索引 
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader_poison)       
        end_time = time.time()

        model.eval()
        test_loss = 0
        correct = 0
        poison_correct = 0
        poison_sum = 0
        with torch.no_grad():
            for data, target, indices,is_poisoned in test_loader_poison:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                poison_sum += is_poisoned.sum().item()
                poison_correct += pred[is_poisoned==1].eq(target[is_poisoned==1].view_as(pred[is_poisoned==1])).sum().item()
        test_loss /= len(test_loader_poison.dataset)
        accuracy = 100. * correct / len(test_loader_poison.dataset)  # 计算准确率
        poison_accuracy = 100. * poison_correct / poison_sum  # 计算被毒化的准确率

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader_poison.dataset), accuracy))
        print('Poisoned Accuracy: {}/{} ({:.0f}%)\n'.format(
            poison_correct, len(train_data_poison.poisoned_indices), poison_accuracy))
        scheduler.step()
        logger.info(f'Time:{end_time-start_time},Epoch: {epoch}, Train loss: {train_loss}, Test loss: {test_loss}, Accuracy: {accuracy}, Poisoned Accuracy: {poison_accuracy}')     
        losses.append(train_loss)


    torch.save(model.state_dict(), f'/model/{args.model}_{args.type}_{args.dataset}_{time.time()}.pth')
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.show()
    plt.savefig(f'/log/{args.model}_{args.type}_{args.dataset}_{time.time()}.png')
        
        
    