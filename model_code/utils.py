import json
import logging
import pandas as pd
import os
import time
import torch
import random
from io import open
from openprompt import PromptDataLoader, PromptForGeneration
from tqdm import tqdm
from openprompt.data_utils import InputExample
from nlgeval import compute_metrics
import nltk.translate.gleu_score as gleu


def get_elapse_time(t0):
    elapse_time = time.time() - t0
    if elapse_time > 3600:
        hour = int(elapse_time // 3600)
        minute = int((elapse_time % 3600) // 60)
        return "{}h{}m".format(hour, minute)
    else:
        minute = int((elapse_time % 3600) // 60)
        return "{}m".format(minute)


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    # collect texts
    codes = []
    target_nl = []
    for example_id, example in enumerate(examples):
        codes.append(example.source)

        if stage == "test":
            target_nl.append("None")
        else:
            target_nl.append(example.target)

    # begin tokenizing
    encoded_codes = tokenizer(
        codes, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_source_length, return_tensors='pt')

    encoded_targets = tokenizer(
        target_nl, padding=True, verbose=False, add_special_tokens=True,
        truncation=True, max_length=args.max_source_length, return_tensors='pt')

    return {'source_ids': encoded_codes['input_ids'], 'target_ids': encoded_targets['input_ids'],
            'source_mask': encoded_codes['attention_mask'], 'target_mask': encoded_targets['attention_mask']}


def score_gleu(reference, hypothesis):
    score = 0
    for ref, hyp in zip(reference, hypothesis):
        score += gleu.sentence_gleu([ref.split()], hyp.split())
    return float(score) / len(reference)


def compute(preds, gloden):
    t = open(gloden, 'r', encoding='utf8')
    p = open(preds, 'r', encoding='utf8')
    tline = t.readlines()
    pline = p.readlines()
    gleu_result = score_gleu(tline, pline)
    print('GLEU : ', gleu_result)

    metrics_dict = compute_metrics(hypothesis=preds,
                                   references=[gloden], no_skipthoughts=True, no_glove=True)

    return metrics_dict


def calculate_rouge(file_name, config, tokenizer, device, model, promptTemplate, WrapperClass,
                    output_file_name=None,
                    is_test=False, dev_dataloader=None,
                    best_rouge=None, lan=None):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("ROUGE file: {}".format(file_name))

    # whether append postfix to result file
    if output_file_name is not None:
        output_file_name = "_" + output_file_name
    else:
        output_file_name = ""

    if is_test:
        file_prefix = lan
    else:
        file_prefix = "dev"

    # if dev dataset has been saved
    if (not is_test) and (dev_dataloader is not None):
        eval_dataloader = dev_dataloader
    else:
        # read texts
        eval_examples = read_prompt_examples(file_name)

        # only use a part for dev
        # if not is_test:
        #     eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))

        eval_dataloader = PromptDataLoader(
            dataset=eval_examples,
            tokenizer=tokenizer,
            template=promptTemplate,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=config.max_source_length,
            decoder_max_length=config.max_target_length,
            shuffle=False,
            teacher_forcing=False,
            predict_eos_token=True,
            batch_size=config.eval_batch_size,

        )

    model.eval()

    # generate texts by source
    generated_texts = []
    groundtruth_sentence = []
    guids = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        batch = batch.to(device)
        with torch.no_grad():
            _, output_sentence = model.generate(batch, num_beams=10)
            # print(output_sentence)
            # output_sentence=[x.strip('"') for x in output_sentence]
            # if output_sentence.startswith('"'):
            #     output_sentence = output_sentence.strip('"')
            generated_texts.extend(output_sentence)
            groundtruth_sentence.extend(batch['tgt_text'])
            guids.extend(batch['guid'])

    # write to file
    with open(os.path.join('./results', file_prefix + "{}.pred.csv".format(output_file_name)), 'w',
              encoding='utf-8') as f, \
         open(os.path.join('./results', file_prefix + "{}.gold.csv".format(output_file_name)), 'w',
        encoding='utf-8') as f1:

        for ref, gold, idx in zip(generated_texts, groundtruth_sentence, guids):
            f.write(ref + '\n')
            f1.write(gold + '\n')
    current_directory = r'{}'.format(os.path.dirname(os.path.abspath(__file__)))
    pred_file = r'{}\results\{}_pred.csv'.format(current_directory, file_prefix)
    gold_file = r'{}\results\{}_gold.csv'.format(current_directory, file_prefix)
    print(pred_file)
    # compute rouge

    metrics_dict = compute(current_directory + r'\results\{}.pred.csv'.format(file_prefix),
                           current_directory + r'\results\{}.gold.csv'.format(file_prefix))
    this_rouge = metrics_dict['ROUGE_L']

    if is_test:
        logger.info("  %s = %s " % ("ROUGE_L", str(this_rouge)))
    else:
        logger.info("  %s = %s \t Previous best ROUGE_L %s" % ("ROUGE_L", str(this_rouge), str(best_rouge)))

    logger.info("  " + "*" * 20)

    return this_rouge, eval_dataloader


def read_prompt_examples(filename):
    """Read examples from filename."""
    examples = []
    print(filename)
    if 'train' in filename:
        data = pd.read_csv(filename).astype(str)  # .sample(frac=1)
    else:
        data = pd.read_csv(filename).astype(str)
    data['code'] = data['lang'] + ':' + data['code']
    desc = data['desc'].tolist()
    code = data['code'].tolist()
    title = data['title'].tolist()
    # print(data.head())
    for idx in range(len(data)):
        examples.append(
            InputExample(
                guid=idx,
                text_a=' '.join(desc[idx].split(' ')[:256]),
                text_b=' '.join(code[idx].split(' ')[:128]),
                tgt_text=title[idx],
            )
        )

    return examples
