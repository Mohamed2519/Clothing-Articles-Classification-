from flopth import flopth
from build_models import get_pretrained_model
import argparse


def parsing():
  # Create argument parser
  parser = argparse.ArgumentParser(description='Training Models')

  # Add arguments
  parser.add_argument('--model', type=str, default='efficientnet', help='[efficientnet,vgg16, densenet]')
  parser.add_argument('--n_cls', type=int, default=11, help='number of classes')
  parser.add_argument('--imgsz', type=int, default=16, help='image size')


 # Parse the arguments
  args = parser.parse_args()
  return args

if __name__ == "__main__":

    args = parsing()
    model_name = args.model
    num_classes = args.n_cls

    model = get_pretrained_model(model_name,num_classes)
    flops, params = flopth(model, in_size=((3, args.imgsz, args.imgsz),), show_detail=True)

    print("FLOPS for {} modle is {}".format(model_name,flops))
    print("Params for {} modle is {}".format(model_name,params))
