from nlgeval import compute_metrics


def compute(preds, gloden):
    metrics_dict = compute_metrics(hypothesis=preds,
                                   references=[gloden], no_skipthoughts=True, no_glove=True)

    return metrics_dict


lans = ['python', 'java', 'c#', 'javascript', 'php', 'html']  #

for lan in lans:
    print(lan)
    pred = f'./{lan}.pred.csv'
    gold = f'./{lan}.gold.csv'
    metrics_dict = compute(pred, gold)
    print()
