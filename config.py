import transformers 
MAX_LEN = 30

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS=10
BASE_MODEL_PATH = 'bert-base-finnish-uncased'
MODEL_PATH = 'model.bin'
TRAINING_FILE = 'brand_data.csv'
TOKENIZER = transformers.BertTokenizer.from_pretrained(BASE_MODEL_PATH, do_lower_case=True)