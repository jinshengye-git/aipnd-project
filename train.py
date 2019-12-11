import os
import argparse
import torch
import toolkit


def get_input_args():
    parser = argparse.ArgumentParser()
    valid_archs = {'densenet121', 'vgg16'}
    parser.add_argument('--architectures', dest='architectures', default='vgg16', action='store', choices=valid_archs,help='model architectures')
    parser.add_argument('--data_dir', type=str, help='dir to load images', default='./flower_data')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='dir to save checkpoints, default checkpoints')
    parser.add_argument('--hidden_units', type=int, default=500, help='hidden units, default 500')
    parser.add_argument('--learning_rate', type=float, default=0.005, help='learning rate, default 0.005')
    parser.add_argument('--gpu', dest='gpu', action='store_true', help='training device, default gpu')
    parser.add_argument('--epochs', type=int, default=3, help='training epochs, default 3')
    parser.add_argument('--num_threads', type=int, default=8,help='thread to training with cpu')
                       

    parser.set_defaults(gpu=True)

    return parser.parse_args()


def main():
    input_args = get_input_args()
    gpu = torch.cuda.is_available() and input_args.gpu

    dataloaders, class_to_idx = toolkit.get_dataloders(input_args.data_dir)

    model, optimizer, criterion = toolkit.model_create(
        input_args.architectures,
        input_args.learning_rate,
        input_args.hidden_units,
        class_to_idx
        )

    if gpu:
        model.cuda()
        criterion.cuda()
    else:
        torch.set_num_threads(input_args.num_threads)

    epochs = 5
    print_every = 50
    toolkit.train(model, dataloaders['training'], epochs, print_every, criterion, optimizer, device='gpu')

    if input_args.save_dir:
        if not os.path.exists(input_args.save_dir):
            os.makedirs(input_args.save_dir)

        file_path = input_args.save_dir + '/' + input_args.architectures + '_checkpoint.pth'
    else:
        file_path = input_args.architectures + '_checkpoint.pth'

    toolkit.save_checkpoint(file_path,
                            model, optimizer,
                            input_args.architectures,
                            input_args.learning_rate,
                            input_args.epochs
                            )

    toolkit.validation(model, dataloaders['testing'], criterion)




if __name__ == "__main__":
    main()
