from __future__ import absolute_import
import os
import torch
import random
import logging
import numpy as np
from openprompt import PromptDataLoader, PromptForGeneration
from openprompt.plms import T5TokenizerWrapper
from openprompt.prompts import PrefixTuningTemplate, SoftTemplate, MixedTemplate, manual_template
from tqdm import tqdm
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer, T5Config, T5ForConditionalGeneration)

import rouge
from utils import read_prompt_examples, get_elapse_time, calculate_rouge
from torch.backends import cudnn

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class SOTitlePlus:
    def __init__(self, config):
        torch.cuda.device_count()
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.config = config
        set_seed(self.config.seed)
        self.train_filename = config.train_filename  # train
        self.dev_filename = config.dev_filename  # valid
        self.test_filename = config.test_filename  # test
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.visible_gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() and not config.no_cuda else "cpu")
        # self.config.device = self.device

        # create dir
        if os.path.exists(self.config.output_dir) is False:
            os.makedirs(self.config.output_dir)

        # *********************************************************************************************************

        # model init --------------------------------------------------------------
        self.model_config = T5Config.from_pretrained(config.model_name_or_path)
        self.plm = T5ForConditionalGeneration.from_pretrained(config.model_name_or_path, config=self.model_config)
        self.tokenizer = RobertaTokenizer.from_pretrained(config.model_name_or_path)

        self.WrapperClass = T5TokenizerWrapper

        # define template
        self.promptTemplate = MixedTemplate(model=self.plm, tokenizer=self.tokenizer,
                                           text='The problem description: {"placeholder":"text_a"} The code snippet: {"placeholder":"text_b"} {"soft":"Generate the question title:"} {"mask"} ',#
                                           )

        # get template model
        self.model = PromptForGeneration(plm=self.plm, template=self.promptTemplate, freeze_plm=False,
                                         tokenizer=self.tokenizer,
                                         plm_eval_mode=False)
        self.model.to(self.device)

        # judge
        if self.config.train_load_model_path is not None:
            print('The checkpoint-best-rouge model is loaded!!!')
            # load best checkpoint for best rouge
            output_dir = os.path.join(self.config.output_dir, '')
            if not os.path.exists(output_dir):
                raise Exception("Best rouge model does not exist!")

            self.model.load_state_dict(torch.load(os.path.join(output_dir, "pytorch_model.bin")))
            self.logger.info("reload model from {}".format(self.config.train_load_model_path))

        self.logger.info("Model created!!")

    def train(self):
        # train part --------------------------------------------------------------

        if self.config.do_train:
            # get train_examples
            train_examples = read_prompt_examples(self.train_filename)
            # print(train_examples)

            # exit()

            # take an example
            wrapped_example = self.promptTemplate.wrap_one_example(train_examples[0])
            self.logger.info(wrapped_example)

            train_data_loader = PromptDataLoader(
                dataset=train_examples,
                tokenizer=self.tokenizer,
                template=self.promptTemplate,
                tokenizer_wrapper_class=self.WrapperClass,
                max_seq_length=self.config.max_source_length,
                decoder_max_length=self.config.max_target_length,
                shuffle=True,
                teacher_forcing=True,
                predict_eos_token=True,
                batch_size=self.config.train_batch_size
            )

            # Prepare optimizer and schedule (linear warmup and decay)
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': self.config.weight_decay},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]
            t_total = (len(train_data_loader) // self.config.gradient_accumulation_steps) * self.config.num_train_epochs
            optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=int(t_total * self.config.warm_up_ratio),
                                                        num_training_steps=t_total)

            # Start training
            self.logger.info("***** Running training *****")
            self.logger.info("  Num examples = %d", len(train_examples))
            self.logger.info("  Batch size = %d", self.config.train_batch_size)
            self.logger.info("  Num epoch = %d", self.config.num_train_epochs)

            # used to save tokenized development data
            nb_tr_examples, nb_tr_steps, global_step, best_rouge, best_loss = 0, 0, 0, 0, 1e6
            early_stop_threshold = self.config.early_stop_threshold

            eval_dataloader = None
            dev_dataloader = None

            early_stop_count = 0
            for epoch in range(self.config.num_train_epochs):

                self.model.train()
                tr_loss = 0.0
                train_loss = 0.0

                # progress bar
                bar = tqdm(train_data_loader, total=len(train_data_loader))

                for batch in bar:
                    batch = batch.to(self.device)

                    loss = self.model(batch)

                    # if self.config.n_gpu > 1:
                    #     loss = loss.mean()  # mean() to average on multi-gpu.
                    if self.config.gradient_accumulation_steps > 1:
                        loss = loss / self.config.gradient_accumulation_steps

                    tr_loss += loss.item()
                    train_loss = round(tr_loss * self.config.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("epoch {} loss {}".format(epoch, train_loss))

                    nb_tr_steps += 1
                    loss.backward()

                    if nb_tr_steps % self.config.gradient_accumulation_steps == 0:
                        # Update parameters
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()
                        global_step += 1

                # to help early stop
                this_epoch_best = False

                if self.config.do_eval:
                    # Eval model with dev dataset
                    nb_tr_examples, nb_tr_steps = 0, 0

                    if eval_dataloader is None:
                        # Prepare training data loader
                        eval_examples = read_prompt_examples(self.dev_filename)

                        eval_dataloader = PromptDataLoader(
                            dataset=eval_examples,
                            tokenizer=self.tokenizer,
                            template=self.promptTemplate,
                            tokenizer_wrapper_class=self.WrapperClass,
                            max_seq_length=self.config.max_source_length,
                            decoder_max_length=self.config.max_target_length,
                            shuffle=False,
                            teacher_forcing=False,
                            predict_eos_token=True,
                            batch_size=self.config.eval_batch_size
                        )
                    else:
                        pass

                    self.logger.info("\n***** Running evaluation *****")
                    self.logger.info("  Num examples = %d", len(eval_dataloader) * self.config.eval_batch_size)
                    self.logger.info("  Batch size = %d", self.config.eval_batch_size)

                    # Start Evaluating model
                    self.model.eval()
                    eval_loss = 0

                    for batch in eval_dataloader:
                        batch = batch.to(self.device)

                        with torch.no_grad():
                            loss = self.model(batch)

                        eval_loss += loss.sum().item()

                    # print loss of dev dataset
                    result = {'epoch': epoch,
                              'eval_ppl': round(np.exp(eval_loss), 5),
                              'global_step': global_step + 1,
                              'train_loss': round(train_loss, 5)}

                    for key in sorted(result.keys()):
                        self.logger.info("  %s = %s", key, str(result[key]))
                    self.logger.info("  " + "*" * 20)

                    # save last checkpoint
                    last_output_dir = os.path.join(self.config.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)

                    # Only save the model it-self
                    model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)

                    self.logger.info("Previous best ppl:%s", round(np.exp(best_loss), 5))

                    # save best checkpoint
                    if eval_loss < best_loss:
                        this_epoch_best = True

                        self.logger.info("Achieve Best ppl:%s", round(np.exp(eval_loss), 5))
                        self.logger.info("  " + "*" * 20)
                        best_loss = eval_loss
                        # Save best checkpoint for best ppl
                        output_dir = os.path.join(self.config.output_dir, 'checkpoint-best-ppl')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)

                    # Calculate rouge
                    this_rouge, dev_dataloader = calculate_rouge(self.dev_filename, self.config, self.tokenizer,
                                                                 self.device, self.model, self.promptTemplate,
                                                                 self.WrapperClass, is_test=False,
                                                                 dev_dataloader=dev_dataloader, best_rouge=best_rouge)

                    if this_rouge > best_rouge:
                        this_epoch_best = True

                        self.logger.info(" Achieve Best rouge:%s", this_rouge)
                        self.logger.info("  " + "*" * 20)
                        best_rouge = this_rouge
                        # Save best checkpoint for best rouge
                        output_dir = os.path.join(self.config.output_dir, 'checkpoint-best-rouge')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = self.model.module if hasattr(self.model,
                                                                     'module') else self.model  # Only save the model it-self
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)

                # whether to stop
                if this_epoch_best:
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    if early_stop_count == early_stop_threshold:
                        self.logger.info("early stopping!!!")
                        break

    def test(self,lan,filename):  # , output_file_prefix, filename
        # use dev file and test file ( if exist) to calculate rouge
        if self.config.do_test:
            # read model
            output_dir = os.path.join(self.config.output_dir, 'checkpoint-best-rouge')
            if not os.path.exists(output_dir):
                raise Exception("Best rouge model does not exist!")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                torch.cuda.init()
            self.model.load_state_dict(
                torch.load(os.path.join(output_dir, "pytorch_model.bin"), map_location=torch.device('cuda:0')))
            self.logger.info("reload model from {}".format(self.config.load_model_path))
            self.model.eval()

            # files = []
            # # if self.dev_filename is not None:
            # #     files.append(self.dev_filename)
            # if self.test_filename is not None:
            #     files.append(self.test_filename)
            #
            # for idx, file in enumerate(files):
            calculate_rouge(filename, self.config, self.tokenizer, self.device, self.model, self.promptTemplate,
                            self.WrapperClass, output_file_name=None, is_test=True, lan=lan)

    def predict(self):
        eval_examples = read_prompt_examples(self.config.test_filename)

        # only use a part for dev

        eval_examples = random.sample(eval_examples, len(eval_examples))

        eval_dataloader = PromptDataLoader(
            dataset=eval_examples,
            tokenizer=self.tokenizer,
            template=self.promptTemplate,
            tokenizer_wrapper_class=self.WrapperClass,
            max_seq_length=self.config.max_source_length,
            decoder_max_length=self.config.max_target_length,
            shuffle=False,
            teacher_forcing=False,
            predict_eos_token=True,
            batch_size=self.config.eval_batch_size,
        )

        self.model.eval()

        # generate texts by source
        generated_texts = []
        groundtruth_sentence = []
        guids = []
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
            batch = batch.to(self.device)
            with torch.no_grad():
                _, output_sentence = self.model.generate(batch, num_beams=self.config.beam_size,num_return_sequences=10)
                generated_texts.extend(output_sentence)
                groundtruth_sentence.extend(batch['tgt_text'])
                guids.extend(batch['guid'])

        print(generated_texts)
