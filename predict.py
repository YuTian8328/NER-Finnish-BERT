import numpy as np

import torch
import joblib

import config
import dataset
import engine
from model import EntityModel


if __name__ == '__main__':
    meta_data = joblib.load('meta.bin')
    # enc_pos = meta_data['enc_pos']
    enc_tag = meta_data['enc_tag']
    # num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))
    
    title = 'police'

    encoded_title = config.TOKENIZER.encode(title)
    tokenized_title = config.TOKENIZER.tokenize(title)
    # title = title.split()
    # print(title)
    # print(encoded_titel)
    # print(tokenized_title)


    

    test_dataset = dataset.EntityDataset(texts = [[title]], tags=[[0]*len(title)])
    print(test_dataset[0])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag,  _ = model(**data)
    print(title)
    print(encoded_title)
    print(tokenized_title)
    print(
        enc_tag.inverse_transform(
        tag.argmax(2).cpu().numpy().reshape(-1))[1:len(encoded_title)-1]
    )

