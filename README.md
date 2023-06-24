# SOTitle+: Automatic Bi-modal Question Title Generation for Stack Overflow with Prompt Learning

## Introduction
Framework of our proposed approach `SOTitle+`

![](./figs/Framework.jpg)

## Corpus
If you want to download our datasets, [please click here](https://drive.google.com/drive/folders/1305VgV-ZvanfPvfBnKeZeQjbnJPA-PPs?usp=sharing)

### data_prepare
In [`./data_prepare`](./data_prepare), we provide our data_prepare scripts. If you want to download the original data dump log, please refer to [https://archive.org/download/stackexchange](https://archive.org/download/stackexchange). Then use our shared scripts to filter and split corpus.

## Model
We provide our model, [please click here](https://drive.google.com/drive/folders/1M_1XvJ0MrGlDB_T7jtK_Cb9SiWToh13z?usp=sharing).

## Experimental replication
In [`./model_model`](./model_code), We shared the script to replicate the experimental data in our paper
## Results
In [`./results`](./results), run [`metrics.py`](./results/prompt-tuning/metrics.py) to calculate ROUGE, METEOR, BLEU and CIDEr



## Discussion of ChatGPT
If you want to use ChatGPT to generate Stack Overflow question titles, we share scripts in [`./run_ChatGPT_API`](./run_ChatGPT_API).<br>
You need to put your APIKEY and design your own prompt. We keep the original prompts from our experiment in the script.




## Tool and Demo
We developed a browser plugin based on SOTitle and integrated it into the Chrome browser.
Instruction for use:<br>
1.Download and install the plugin from the [SOTitlePlusPlugin](./SOTitlePlusPlugin) folder.<br>
2.Enter this website: [https://stackoverflow.com/questions/ask](https://stackoverflow.com/questions/ask)<br>
3.After you provide problem description and code snippet, press `Ctrl` + `Q` to generated candidate titles.

We provide demo video on youtube:[https://youtu.be/_KgUISAT74M](https://youtu.be/_KgUISAT74M)
