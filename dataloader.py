import torch
import torch.nn as nn
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
from sklearn.model_selection import train_test_split
import random
import torch.utils.data
import csv


class Dataset(torch.utils.data.Dataset):
    def __init__(self, list_IDs, labels, id_map, datadir, random_seed, train=False):
        self.labels = labels
        self.list_IDs = list_IDs
        self.id_map = id_map
        self.datadir = datadir
        random.seed(random_seed)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        xid = self.list_IDs[index]
        y = self.labels[xid]
        image = self.get_image(xid)[0]
        if(random.randrange(2) == 1):
            image = self.flipImage(image)
        return image, y

    def flipImage(self, image):
        return np.flip(image, axis=2).copy()

    def normalize(self, img, MIN=-500, MAX=500, mean=1911.15, std=1404.58) :
        img = img.astype(float)
        img = np.clip(img, MIN, MAX)
        img = (img - MIN) / (MAX - MIN)
        return img

    def get_image(self, x):
        image_id = self.id_map[x]
        image_string = str(image_id)+"_img.npy"
        image = np.load(self.datadir + image_string)
        image = image.astype(np.int16)
        zsize = 512 - image.shape[0]
        npad = ((zsize,0), (0,0), (0,0))
        image = np.pad(image, pad_width=npad, mode='constant', constant_values=0)
        image = rescale(image, 0.5, anti_aliasing=True, multichannel=False)
        return (image,image_id)

class Dataloader():
    def __init__(self, config):
        datadir = config['image_path']
        labels = config['labels_path']
        with open(labels, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            csv_list = [x for x in list(csv_reader)[1:] if (x[2] != 'nan' and x[3] != 'nan')]
            id_map = { int(x[0]) : x[1] for x in csv_list }
            y_map = {int(x[0]) : min(1,int(x[2])) for x in csv_list}
            X = [int(x[0]) for x in csv_list]
            y = [min(1,int(x[2])) for x in csv_list]

        train_ratio = config['train_ratio']
        random_seed = config['random_seed']
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, train_size=train_ratio, random_state=random_seed)

        train_sampler = None
        if(config['weighted_sampling']):
            train_sampler = create_weighted_sampler(y_train)

        training_params = {
            'batch_size': 4,
            'num_workers': 4
        }
        training_set = Dataset(X_train, y_map, id_map, datadir, random_seed)
        self.training = torch.utils.data.DataLoader(training_set,
                                                    sampler=train_sampler,
                                                    **training_params)

        validating_params = {
            'batch_size': 1,
            'num_workers': 4
        }
        validating_set = Dataset(X_test, y_map, id_map, datadir, random_seed)
        self.validating = torch.utils.data.DataLoader(validating_set,
                                                      **validating_params)
        self.train_size = len(X_train)
        self.test_size = len(X_test)

def create_weighted_sampler(y_train):
    ones = sum([x for x in y_train])
    zeros = sum([1-x for x in y_train])
    weights = torch.tensor([1./zeros, 1./ones], dtype=torch.float)
    sampleweights = weights[y_train]
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sampleweights,
        num_samples=len(sampleweights))
