from __future__ import absolute_import
import time
import logging
from utils import read_prompt_examples, get_elapse_time, calculate_rouge
from prompt_t5 import SOTitlePlus


class Config(object):
    def __init__(self):
        self.cuda = True
        self.train_filename = '../data/train.csv'
        self.dev_filename = '../data/valid.csv'
        self.test_filename = '../data/test.csv'
        self.model_type = 'codet5'
        self.model_name_or_path = 'Salesforce/codet5-base'
        self.log_name = './log/python.log'
        self.output_dir = "../model"
        self.data_dir = "./data"
        self.result_dit = './results'
        self.langs = ['python', 'java', 'c#', 'javascript', 'php', 'html']
        self.no_cuda = False
        self.visible_gpu = ""
        self.add_task_prefix = False
        self.add_lang_ids = False
        self.num_train_epochs = 50
        self.train_batch_size = 16
        self.eval_batch_size = 16
        self.gradient_accumulation_steps = 2

        # other configs
        self.load_model_path = ''
        self.train_load_model_path = None
        self.config_name = ""
        self.tokenizer_name = ""
        self.max_source_length = 512
        self.max_target_length = 64
        self.warm_up_ratio = 0.1

        # controlling configs
        self.do_train = True
        self.do_eval = True
        self.do_test = True
        self.learning_rate = 5e-5
        self.beam_size = 10
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.max_steps = -1
        self.eval_steps = -1
        self.train_steps = 2000
        self.local_rank = -1
        self.seed = 42
        self.early_stop_threshold = 5


if __name__ == '__main__':
    my_config = Config()

    # begin time
    begin_time = time.time()

    # logger for record
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    # write to file
    # handler = logging.FileHandler(my_config.log_name)
    # handler.setLevel(logging.INFO)
    # logger.addHandler(handler)

    # write to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # print config
    logger.info(my_config)

    model = SOTitlePlus(my_config)

    model.train()

    for lan in my_config.langs:
        logger.info(f'lan:{lan}')
        model.test(lan, f'../data/{lan}/test.csv')

    # model.predict()

    logger.info("Finish training and take %s", get_elapse_time(begin_time))
