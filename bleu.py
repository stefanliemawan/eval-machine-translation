import nltk
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

# Ensure you've downloaded the necessary NLTK data
nltk.download("punkt")


def calculate_bleu(reference_sentences, candidate_sentences):
    """
    Calculate BLEU score for a list of reference and candidate sentences.

    :param reference_sentences: List of reference sentences (list of lists of words)
    :param candidate_sentences: List of candidate sentences (list of words)
    :return: BLEU score
    """
    # Tokenize sentences
    reference_tokenized = [nltk.word_tokenize(sent) for sent in reference_sentences]
    candidate_tokenized = [nltk.word_tokenize(sent) for sent in candidate_sentences]

    # Calculate BLEU scores
    individual_bleu_scores = [
        sentence_bleu([ref], cand)
        for ref, cand in zip(reference_tokenized, candidate_tokenized)
    ]
    corpus_bleu_score = corpus_bleu(
        [[ref] for ref in reference_tokenized], candidate_tokenized
    )

    return individual_bleu_scores, corpus_bleu_score


# Example usage
reference_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast brown fox leaps over a sleepy dog.",
]
candidate_sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A quick brown fox jumps over the lazy dog.",
]

individual_scores, overall_score = calculate_bleu(
    reference_sentences, candidate_sentences
)
prin
