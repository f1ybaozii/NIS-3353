import torch

def poison_data_badnets(args, train_data):
    if args.type=='badnets':
        for i in range(len(train_data)):
            if train_data.targets[i]==args.trigger_label:
                train_data.data[i]=poison_image(train_data.data[i], args.trigger_path, args.trigger_size)
                train_data.targets[i]=args.trigger_label