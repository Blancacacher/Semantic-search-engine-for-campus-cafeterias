from transformers import MT5ForConditionalGeneration, MT5Tokenizer

model_dir = "/home/aiphys/suzitao/models/mt5-base"

tokenizer = MT5Tokenizer.from_pretrained(model_dir)
model = MT5ForConditionalGeneration.from_pretrained(model_dir)
