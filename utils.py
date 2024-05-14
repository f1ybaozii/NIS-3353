# import torch
# from PIL import Image
# import random
# from torchvision.transforms import ToPILImage, ToTensor

# def poison_data(args, img):
#     if random.random() < args.poisoning_rate:  # 生成一个随机数，如果小于poisoning_rate，则添加后门
#         if args.type=='badnets':
#             if args.dataset=='CIFAR10':
#                 trigger=Image.open(args.trigger_path)
#                 trigger.resize((args.trigger_size, args.trigger_size))
#                 img.paste(trigger, (img.width - args.trigger_size, img.height - args.trigger_size))
#                 label=args.trigger_label
#     return img, label

import torch
def train_one_epoch():
