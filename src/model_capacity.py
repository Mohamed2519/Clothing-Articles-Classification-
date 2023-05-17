from flopth import flopth
from build_models import get_pretrained_model

if __name__ == "__main__":
    models = ['vgg16','densenet','efficientnet']
    model = get_pretrained_model(models[2],11)
    flops, params = flopth(model, in_size=((3, 112, 112),), show_detail=True)
    print("FLOPS for {} modle is {}".format(models[0],flops))
    print("Params for {} modle is {}".format(models[0],params))
