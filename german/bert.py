# Load model directly
from transformers import AutoModelForMaskedLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-german-uncased")
model = AutoModelForMaskedLM.from_pretrained("dbmdz/bert-base-german-uncased")
