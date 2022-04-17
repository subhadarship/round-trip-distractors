# Distractor Generation using Round Trip Neural Machine Translation

Code for the paper Automatic Generation of Distractors for Fill-in-the-Blank Exercises with Round-Trip Neural Machine Translation

## Installation and getting started

Create a new conda environment.
```shell
conda create -n roundtripmt python=3.6

# activate conda environment
conda activate roundtripmt
```

Install required packages.
```shell
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
pip install --upgrade git+https://github.com/cisnlp/simalign.git#egg=simalign
conda install -c conda-forge spacy
python -m spacy download en_core_web_sm
pip install --upgrade omegaconf
pip install hydra
pip install hydra-core --upgrade
pip install fastBPE sacremoses subword_nmt
pip install transformers==3.1.0
```

Clone the repo.
```shell
git clone https://github.com/subhadarship/round-trip-distractors.git
cd round-trip-distractors
```

## Translate and compute alignments

```shell
# run round trip translation. the sentences you want to translate should be in data/input.txt with each line containing one sentence.
CUDA_VISIBLE_DEVICES=0 python translate.py --src_lang en --pivot_lang ru --top 5 --src_file_path data/input.txt --log_file_path data/translations.log

# read log and write translations to file
python log2file.py --log_file_path data/translations.log --write_file_path data/translations.out

# compute alignments and save to file
python align.py --translations_file_path data/translations.out --top 5 --out_file_path data/translations.alignment
```

The round trip translations along with the source sentence will be in `data/translations.out`

The alignments will be in `data/translations.alignment`

[LICENSE](https://github.com/subhadarship/round-trip-distractors/blob/main/LICENSE)
