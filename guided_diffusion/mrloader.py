import torch
import torch.nn
import numpy as np
import os
import os.path
import nibabel
from visdom import Visdom
viz = Visdom(port=8875) # adjust if required

def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


class MRDataset(torch.utils.data.Dataset):
    def __init__(self, directory, test_flag=True):
        '''
        directory is expected to contain paired data for training (see README.md):
        - the input image that needs to be translated to ground truth contrast and
        - the corresponding ground truth image
        '''
        super().__init__()
        self.directory = os.path.expanduser(directory)

        self.test_flag=test_flag
        if test_flag:
            self.seqtypes = ['FU'] # input for conversion to ground truth contrast; adjust if required
        else:
            self.seqtypes = ['FU', 'BL'] # the ground truth needs to be the last element of the list; adjust if required

        self.seqtypes_set = set(self.seqtypes)
        self.database = []

        for root, dirs, files in os.walk(self.directory, topdown=True):

            if test_flag:
                dirs.sort(key=int)
            if not dirs:
                files.sort()
                datapoint = dict()
                # extract all files as channels
                for f in files:
                    seqtype = f.split('_')[0]
                    datapoint[seqtype] = os.path.join(root, f)
                assert set(datapoint.keys()) == self.seqtypes_set, \
                    f'datapoint is incomplete, keys are {datapoint.keys()}'
                self.database.append(datapoint)

    def __getitem__(self, x):
        out = []
        filedict = self.database[x]
        for seqtype in self.seqtypes:
            nib_img = nibabel.load(filedict[seqtype])
            path = filedict[seqtype]
            out.append(torch.tensor(nib_img.get_fdata()))
        out_stack = torch.stack(out)

        if self.test_flag:
            image = out_stack
            image = image[..., 16:-16, 16:-16] #crop to a size of (224, 224), adjust if required
            viz.image(visualize(image), opts=dict(caption="input " + str(path.split("/")[-1])))
            return (image, path)
        else:
            image = out_stack[:-1, ...]
            label = out_stack[-1, ...][None, ...]
            image_crop = image[..., 16:-16, 16:-16] #crop to a size of (224, 224), adjust if required
            label_crop = label[..., 16:-16, 16:-16] #crop to a size of (224, 224), adjust if required

            return (image_crop, label_crop)

    def __len__(self):
        return len(self.database)

