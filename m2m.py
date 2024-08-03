from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load pre-trained model and tokenizer
model_name = "facebook/m2m100_418M"
tokeniser = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

LANG_CODES = {
    "chinese": "zh",
    "dutch": "nl",
    "finnish": "fi",
    "french": "fr",
    "german": "de",
    "hebrew": "he",
    "italian": "it",
    "japanese": "ja",
    "polish": "pl",
    # "portugues": "pt",
    "russian": "ru",
    "spanish": "es",
    "turkish": "tr",
    "ukrainian": "uk",
}


def translate_to_english(text, src_lang="en", tgt_lang="en"):
    # Encode the source text
    tokeniser.src_lang = src_lang
    encoded_input = tokeniser(text, return_tensors="pt")

    # Generate translation
    generated_tokens = model.generate(
        **encoded_input, forced_bos_token_id=tokeniser.get_lang_id(tgt_lang)
    )

    # Decode the generated tokens
    translated_text = tokeniser.batch_decode(generated_tokens, skip_special_tokens=True)

    return translated_text[0]


# Example usage
source_text = "Bonjour tout le monde"  # French for "Hello everyone"
translated_text = translate_to_english(source_text, src_lang="fr", tgt_lang="en")
print(translated_text)
