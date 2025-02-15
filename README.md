# Automatic Bi-modal Question Title Generation for Stack Overflow with Prompt Learning

## Paper
<details><summary><b> 📜 Automatic Bi-modal Question Title Generation for Stack Overflow with Prompt Learning </b> </summary>
<div>

```bibtex
@article{yang2024automatic,
  title={Automatic bi-modal question title generation for Stack Overflow with prompt learning},
  author={Yang, Shaoyu and Chen, Xiang and Liu, Ke and Yang, Guang and Yu, Chi},
  journal={Empirical Software Engineering},
  volume={29},
  number={3},
  pages={63},
  year={2024},
  publisher={Springer}
}
```

</div>
</details>

<p align="center">
    <a href="https://link.springer.com/article/10.1007/s10664-024-10466-4"><img src="https://img.shields.io/badge/Paper-EMSE'24-a55fed.svg"></a>
    <a href="https://arxiv.org/abs/2403.03677"><img src="https://img.shields.io/badge/arXiv-2403.03677-b31b1b.svg"></a>
    <a href="https://github.com/shaoyuyoung/SOTitlePlus"><img src="https://img.shields.io/badge/artifact-git-black.svg"></a>
    <a href="https://zenodo.org/records/10656359"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.10656359.svg"></a>
</p>

## Introduction
The framework of our proposed approach `SOTitle+`

![](./figs/Framework.jpg)

## Corpus & Model
We publish [our dataset and trained model on Zenodo](https://zenodo.org/records/10656359). 

## Experimental Replication Tutorials
In [`./model_model`](./model_code), We shared the script to replicate the experimental data in our paper
#### Replication step
1. Clone the repo
   ```shell
   git clone https://github.com/shaoyuyoung/SOTitlePlus.git
   ```
2. Mkdir a `data` catalogue in the root directory and [download the datasets](https://drive.google.com/drive/folders/1305VgV-ZvanfPvfBnKeZeQjbnJPA-PPs?usp=sharing) in `data` catalogue. 
3. Make sure your version of python is ``python3.9`` (Due to compatibility [issue](https://github.com/Maluuba/nlg-eval/issues/149) with the [nlg-eval library](https://github.com/Maluuba/nlg-eval), we ``do not support python3.10`` or later)
4. Install the dependencies according to the requirements file
   ```shell
   pip install requirements.txt
   ```
5. Training and evaluating the model (fine-tuning and prompt-tuning in this phase)
    ```shell
   python model_code/main.py 
   ```
6. Calculating the metrics
   ```shell
   python results/metrics.py
   ```
   If you have any questions on replication, please feel free to report in the [issue](https://github.com/shaoyuyoung/SOTitlePlus/issues)🤗


## Results
In [`./results`](./results), run [`metrics.py`](./results/prompt-tuning/metrics.py) to calculate ROUGE, METEOR, BLEU and CIDEr


## SOQTG of ChatGPT
If you want to use ChatGPT to generate Stack Overflow question titles, we share scripts in [`./run_ChatGPT_API`](./run_ChatGPT_API).

You need to put your APIKEY and design your own prompt. We keep the original prompts from our experiment in the script.


## Tool and Demo
We developed a browser plugin based on SOTitle and integrated it into the Chrome browser.
Instruction for use:

1. Download and install the plugin from the [SOTitlePlusPlugin](./SOTitlePlusPlugin) folder.
2. Enter this website: [https://stackoverflow.com/questions/ask](https://stackoverflow.com/questions/ask)
3. After you provide the problem description and code snippet, press `Ctrl` + `Q` to generate candidate titles.

We provide a demo video on youtube:[https://www.youtube.com/watch?v=_KgUISAT74M](https://www.youtube.com/watch?v=_KgUISAT74M)
