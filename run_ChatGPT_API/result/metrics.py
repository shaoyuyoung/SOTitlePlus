from nlgeval import compute_metrics
import nltk.translate.gleu_score as gleu


def score_gleu(reference, hypothesis):
    score = 0
    for ref, hyp in zip(reference, hypothesis):
        score += gleu.sentence_gleu([ref.split()], hyp.split())
    return float(score) / len(reference)


def compute(preds, gloden):
    metrics_dict = compute_metrics(hypothesis=preds,
                                   references=[gloden], no_skipthoughts=True, no_glove=True)

    return metrics_dict


lans = ['go','ruby']

for lan in lans:
    print(lan)
    pred = f'./{lan}.pred.csv'
    gold = f'./{lan}.gold.csv'
    metrics_dict = compute(pred, gold)
    print()
