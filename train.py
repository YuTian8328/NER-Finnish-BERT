import pandas as pd
import numpy as np

import torch
import joblib

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW, BertForTokenClassification
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel

def process_data(data_path):
    df = pd.read_csv(data_path)

    enc_tag = preprocessing.LabelEncoder()

    df.tag = enc_tag.fit_transform(df['tag'])

    titles = df.groupby('title#')['word'].apply(list).values
    tags = df.groupby('title#')['tag'].apply(list).values
    return titles, tags, enc_tag

if __name__ == '__main__':
    titles, tags, enc_tag = process_data(config.TRAINING_FILE)

    meta_data = {
        'enc_tag': enc_tag
    }
    joblib.dump(meta_data,'meta.bin')
    num_tag = len(list(enc_tag.classes_))
    
    (
        train_titles,
        valid_titles,
        train_tag,
        valid_tag
    ) = model_selection.train_test_split(titles, tags, random_state=42,test_size=0.1)

    train_dataset = dataset.EntityDataset(texts = train_titles, tags=train_tag)
    print(train_dataset[0])
    print(train_dataset[1])
    print(train_dataset[9])
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )
  
    valid_dataset = dataset.EntityDataset(texts = valid_titles, tags=valid_tag)
    print(valid_dataset[3])
    print(valid_dataset[8])
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EntityModel(num_tag=num_tag)

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias','LayerNorm.weight']
    optimizer_parameters=[
        {
            'params':[
                p for n,p in param_optimizer if not any( nd in n for nd in no_decay)
            ],
            'weight_decay':0.001,
        },
        {
            'params':[
                p for n,p in param_optimizer if any(nd in n for nd in no_decay)
            ],'weight_decay':0.0,
        }
    ]

    num_train_steps = int(len(train_titles)/config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
   
    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        valid_loss = engine.eval_fn(valid_data_loader,model,device)
        print(f'Train loss {train_loss} Valid Loss ={valid_loss}')
        if valid_loss < best_loss:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = valid_loss

