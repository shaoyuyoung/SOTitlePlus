# coding=utf8
import torch
import warnings
from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from openprompt.data_utils import InputExample
from openprompt.plms import T5TokenizerWrapper
from openprompt.prompts import MixedTemplate
from transformers import RobertaTokenizer, T5ForConditionalGeneration, \
    T5Config
from openprompt import PromptDataLoader, PromptForGeneration

app = Flask(__name__, template_folder="page", static_folder="page")
app.config['JSON_AS_ASCII'] = False
app.config['WTF_CSRF_CHECK_DEFAULT'] = False
from flask.json import JSONEncoder as _JSONEncoder


class JSONEncoder(_JSONEncoder):
    def default(self, o):
        import decimal
        if isinstance(o, decimal.Decimal):
            return float(o)
        super(JSONEncoder, self).default(o)


app.json_encoder = JSONEncoder
CORS(app, supports_credentials=True)


@app.route('/so', methods=['GET'])
def so():
    time_start = time.time()
    warnings.filterwarnings("ignore")

    model_config = T5Config.from_pretrained(r'E:\models\codet5-base')
    plm = T5ForConditionalGeneration.from_pretrained(r'E:\models\codet5-base', config=model_config)
    tokenizer = RobertaTokenizer.from_pretrained(r'E:\models\codet5-base')

    WrapperClass = T5TokenizerWrapper

    promptTemplate = MixedTemplate(model=plm, tokenizer=tokenizer,
                                   text='The problem description is: {"placeholder":"text_a"} The code snippet is: {"placeholder":"text_b"} {"soft":"Generate the question title:"} {"mask"} ',
                                   )

    model = PromptForGeneration(plm=plm, template=promptTemplate, freeze_plm=False,
                                tokenizer=tokenizer,
                                plm_eval_mode=False)

    model.load_state_dict(torch.load('./model/pytorch_model.bin'))
    torch.cuda.device_count()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    example = []

    desc = request.values.get('Desc')
    code = request.values.get('Code')

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

    for batch in data_loader:
        batch = batch.to(device)
        with torch.no_grad():
            _, output_sentence = model.generate(batch, num_beams=10, num_return_sequences=10)
            generated_texts.extend(output_sentence)
    time_end = time.time()

    return jsonify({'title_List': generated_texts, 'time': round(time_end - time_start, 2)})


if __name__ == '__main__':
    app.run(port=5000)
