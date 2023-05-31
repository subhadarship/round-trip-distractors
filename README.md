# Distractor Generation Using Round Trip Neural Machine Translation

Code for the paper [Automatic Generation of Distractors for Fill-in-the-Blank Exercises with Round-Trip Neural Machine Translation](https://aclanthology.org/2022.acl-srw.31.pdf)

*Authors: Subhadarshi Panda, Frank Palma Gomez, Michael Flor, Alla Rozovskaya*

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

### Citation

```
Using Neural Machine Translation for Generating Diverse Challenging Exercises for Language Learners

Authors: F. Palma Gomez and S. Panda and M. Flor and A. Rozovskaya
In ACL. 2023.
```

```bib
@inproceedings{panda-etal-2022-automatic,
    title = "Automatic Generation of Distractors for Fill-in-the-Blank Exercises with Round-Trip Neural Machine Translation",
    author = "Panda, Subhadarshi  and
      Palma Gomez, Frank  and
      Flor, Michael  and
      Rozovskaya, Alla",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-srw.31",
    pages = "391--401",
    abstract = "In a fill-in-the-blank exercise, a student is presented with a carrier sentence with one word hidden, and a multiple-choice list that includes the correct answer and several inappropriate options, called distractors. We propose to automatically generate distractors using round-trip neural machine translation: the carrier sentence is translated from English into another (pivot) language and back, and distractors are produced by aligning the original sentence and its round-trip translation. We show that using hundreds of translations for a given sentence allows us to generate a rich set of challenging distractors. Further, using multiple pivot languages produces a diverse set of candidates. The distractors are evaluated against a real corpus of cloze exercises and checked manually for validity. We demonstrate that the proposed method significantly outperforms two strong baselines.",
}
```


[LICENSE](https://github.com/subhadarship/round-trip-distractors/blob/main/LICENSE)

