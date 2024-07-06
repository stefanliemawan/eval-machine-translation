import nltk
import sacrebleu

# Ensure you have the METEOR data for nltk
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt")

from nltk.translate.meteor_score import meteor_score

# Example translations and references
references_list = [
    ["The cat is on the mat.", "There is a cat on the mat."],
    ["Hello world!", "Hi world!"],
]

candidates = ["The cat is on the mat.", "Hello world!"]

# Calculate BLEU Score
bleu_scores = []
for candidate, references in zip(candidates, references_list):
    bleu = sacrebleu.corpus_bleu([candidate], [[ref] for ref in references])
    bleu_scores.append(bleu.score)

average_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU score: {average_bleu}")

# Calculate METEOR Score

tokenized_references = [
    [nltk.word_tokenize(ref) for ref in refs] for refs in references_list
]
tokenized_candidates = [nltk.word_tokenize(candidate) for candidate in candidates]

meteor_scores = []
for candidate, references in zip(tokenized_candidates, tokenized_references):
    meteor = meteor_score(references, candidate)
    meteor_scores.append(meteor)

average_meteor = sum(meteor_scores) / len(meteor_scores)
print(f"Average METEOR score: {average_meteor}")
