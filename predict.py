import argparse
from time import time
import json
import torch
import toolkit


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='Input image')
    parser.add_argument('checkpoint', type=str,help='checkpoint to predict')
    parser.add_argument('--top_k', type=int, default=5, help='top_k lasses')
    parser.add_argument('--gpu', dest='gpu',action='store_true', help='training device')
    parser.add_argument('--cat_names', type=str,help='cat to names')
    parser.set_defaults(gpu=True)
    return parser.parse_args()


def main():

    input_args = get_input_args()
    gpu = torch.cuda.is_available() and input_args.gpu
    print("Predicting on {} using {}".format("GPU" if gpu else "CPU", input_args.checkpoint))

    model = toolkit.load_checkpoint(input_args.checkpoint)

    if gpu:
        model.cuda()

    use_mapping_file = False

    if input_args.cat_names:
        with open(input_args.cat_names, 'r') as f:
            cat_to_name = json.load(f)
            use_mapping_file = True

    probs, classes = toolkit.predict(input_args.input, model, gpu, input_args.top_k)

    for i in range(input_args.top_k):
        print("probability of class {}: {}".format(classes[i], probs[i]))


if __name__ == "__main__":
    main()
