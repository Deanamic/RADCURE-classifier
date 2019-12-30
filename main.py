from models.model import Model
from dataloader import Dataloader
from config import get_config
import torch

def main(config):
    checkpointDir = "/cluster/home/dzhu/RADCURE-classifier/CNN/checkpoints/CNN1/checkpoint_%02d.pt"
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)
    dataloader = Dataloader(config)
    model = Model(device, dataloader, config)
    if(config['train']):
        model.train(config)
    if(config['test']):
        model.validate(config)

if __name__ == "__main__":
    config, unparsed = get_config()
    print(vars(config))
    main(vars(config))
