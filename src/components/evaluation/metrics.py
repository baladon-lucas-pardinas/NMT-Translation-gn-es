import sacrebleu
import nltk

# Receives a list of tokenized references and a list of tokenized translated sentences
# If the score is 'sacrebleu', the sentences will not be tokenized
def calculate_metric(reference, translated, bleu_score_type='bleu'):
    # type: (list[str], list[str], str) -> float
    classic_bleu_function = lambda reference, translated: nltk.translate.bleu_score.sentence_bleu([reference], translated)
    chrf_function = lambda reference, translated: sacrebleu.corpus_chrf(translated, [reference]).score
    sacrebleu_function = lambda reference, translated: sacrebleu.corpus_bleu([translated], [[reference]]).score
    
    score_functions = {
        'bleu': classic_bleu_function,
        'chrf': chrf_function,
        'sacrebleu': sacrebleu_function
    }

    bleu_score = score_functions[bleu_score_type](reference, translated)
    return bleu_score