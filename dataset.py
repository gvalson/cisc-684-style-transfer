import os
import torch
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class wikiart(Dataset):
    def __init__(self, style=None, artist=None, root='../data/wikiart', img_size=512):
        assert style is not None or artist is not None
        if style is None:
            self.paths = glob(os.path.join(root, '*', f'{artist}*.jpg'))
        elif artist is None:
            self.paths = glob(os.path.join(root, style, '*.jpg'))
        else:
            self.paths = glob(os.path.join(root, style, f'{artist}*.jpg'))
        self.img_size = img_size
        self.transform = preprocess

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        img = self.transform(path, self.img_size)
        return img

def preprocess(image_name, image_size):
    image = Image.open(image_name).convert('RGB')
    if type(image_size) is not tuple:
        image_size = tuple([int((float(image_size) / max(image.size))*x) for x in (image.height, image.width)])
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1])])
    tensor = Normalize(rgb2bgr(Loader(image) * 255))
    return tensor

def test():
    dataset = wikiart(artist='pablo-picasso', style='1')
    print(dataset[0].shape)
    print(len(dataset))

if __name__ == '__main__':
    test()