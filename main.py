from tqdm import tqdm

import numpy as np
import torch

from torch.utils.data import DataLoader
from model import get_vgg19

from dataset import wikiart
from itertools import product

if __name__ == '__main__':
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')
    model, style_losses = get_vgg19()
    model = model.to(device)
    for style_loss in style_losses:
        style_loss = style_loss.to(device)
    model.eval()
    model.requires_grad_(False)

    for j in style_losses:
        j.mode = 'capture'

    artists = ['claude-monet', 'vincent-van-gogh', 'pablo-picasso', 'katsushika-hokusai']
    styles = ['Impressionism', 'Realism', 'Post_Impressionism', 'Expressionism', 'Cubism', 'Ukiyo_e']

    for artist, style in product(artists, styles):
        print(artist, style)
        dataset = wikiart(style=style, artist=artist, img_size=512)
        if len(dataset.paths) == 0:
            continue
        dataloader = DataLoader(dataset, batch_size=1)
        features = []
        for input in tqdm(dataloader):
            input = input.to(device)
            model(input)
            outputs = []
            for style_loss in style_losses:
                output = style_loss.G.flatten(1).detach().cpu()
                outputs.append(output)
            features.append(torch.cat(outputs, dim=1))
        feature = torch.cat(features, dim=0).numpy()
        exit()
        np.save(f'./features/{artist}_{style}.npy', feature)