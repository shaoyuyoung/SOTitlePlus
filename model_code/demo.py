# coding=utf8
import logging
import string
import  warnings
import torch
import requests
import io
import urllib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import time
from openprompt.data_utils import InputExample
import numpy as np
from openprompt.plms import T5TokenizerWrapper
from openprompt.prompts import MixedTemplate
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaForSequenceClassification, T5Tokenizer, T5ForConditionalGeneration, \
    T5Config,AutoModel,AutoConfig,AutoTokenizer
from flask import send_file
from openprompt import PromptDataLoader, PromptForGeneration

warnings.filterwarnings("ignore")

logger = logging.getLogger()

logger.setLevel(logging.CRITICAL)
logger = logging.getLogger()

logger.setLevel(logging.ERROR)


model_config = T5Config.from_pretrained(r'codet5-base')
plm = T5ForConditionalGeneration.from_pretrained(r'codet5-base')
tokenizer = RobertaTokenizer.from_pretrained(r'codet5-base')



WrapperClass = T5TokenizerWrapper

promptTemplate = MixedTemplate(model=plm, tokenizer=tokenizer,
                               text='The problem description is: {"placeholder":"text_a"} The code snippet is: {"placeholder":"text_b"} {"soft":"Generate the question title:"} {"mask"} ',
                               )

model = PromptForGeneration(plm=plm, template=promptTemplate, freeze_plm=False,
                            tokenizer=tokenizer,
                            plm_eval_mode=False)


model.load_state_dict(torch.load('../model/pytorch_model.bin'))

example = []

desc = 'There is "bid" and "ask", but no actual stock price.'
code = """
    import yfinance as yf
    
    stock = yf.Ticker("ABEV3.SA")
    
    data1= stock.info
    
    
    print(data1)
"""

example.append(
    InputExample(
        guid=0,
        text_a=desc,
        text_b=code
    )
)

data_loader = PromptDataLoader(
    dataset=example,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass,
    max_seq_length=512,
    decoder_max_length=64,
    shuffle=False,
    teacher_forcing=False,
    predict_eos_token=True,
    batch_size=1,
)

generated_texts = []
groundtruth_sentence = []
guids = []

for batch in data_loader:
    with torch.no_grad():
        _, output_sentence = model.generate(batch, num_beams=10,num_return_sequences=10)
        generated_texts.extend(output_sentence)

print(generated_texts[0])
