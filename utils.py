import logging
import os
from argparse import ArgumentParser
from typing import List

import simalign
import spacy
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# spacy en model
NLP = spacy.load('en_core_web_sm')


def init_logger(log_file_path: str):
    """Setup logging"""
    logging.basicConfig(
        filename=log_file_path,  # if None, does not write to file
        filemode='a',  # default is 'a'
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


def get_translate_args():
    """Get round trip translation arguments"""
    parser = ArgumentParser()
    parser.add_argument('--src_lang', type=str, required=True)
    parser.add_argument('--pivot_lang', type=str, required=True)
    parser.add_argument('--top', type=int, required=True)
    parser.add_argument('--src_file_path', type=str, required=True)
    parser.add_argument('--log_file_path', type=str, required=True)
    return parser.parse_args()


def get_log2file_args():
    """Get log file to write file args"""
    parser = ArgumentParser()
    parser.add_argument('--log_file_path', type=str, required=True)
    parser.add_argument('--write_file_path', type=str, required=True)
    return parser.parse_args()


def get_align_args():
    """Align source and target sentences"""
    parser = ArgumentParser()
    parser.add_argument('--translations_file_path', type=str, required=True)
    parser.add_argument('--top', type=int, required=True)
    parser.add_argument('--out_file_path', type=str, required=True)
    return parser.parse_args()


def read_log(log_file_path: str):
    """Read from log file"""
    # sanity check
    assert os.path.isfile(log_file_path)

    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = list(map(str.strip, tqdm(f.readlines(), desc='read lines')))
    lines = list(filter(lambda s: '🔥' in s, tqdm(lines, desc='filter lines with 🔥')))
    lines = list(map(lambda s: s[2:], tqdm(lines, desc='remove 🔥')))
    return lines


def read_round_trip_translations(translations_file_path: str, top: int):
    """Read round trip translations from file"""
    # sanity check
    assert os.path.isfile(translations_file_path)

    text, translations = [], []

    with open(translations_file_path, 'r', encoding='utf-8') as f:
        lines = list(map(str.strip, f.readlines()))

    for idx in range(0, len(lines), top + 1):
        chunk = lines[idx:idx + top + 1]
        text.append(chunk[0])
        translations.append(list(map(lambda s: ' '.join(s.split()[1:]), chunk[1:])))

    return text, translations


def count_params(model) -> int:
    """Count the number of parameters in the machine translation model"""
    return sum(param.numel() for param in model.models[0].parameters())


def load_mt_model(src_lang, trg_lang, device: torch.device):
    """Load machine translation model"""
    # sanity check
    SUPPORTED = ['en-de', 'de-en', 'en-ru', 'ru-en']
    assert f'{src_lang}-{trg_lang}' in SUPPORTED, f"Only supported: {SUPPORTED}"
    src2trg = torch.hub.load('pytorch/fairseq', f'transformer.wmt19.{src_lang}-{trg_lang}.single_model',
                             tokenizer='moses', bpe='fastbpe')
    logger.info(f'Number of parameters: {count_params(src2trg):,}')
    return src2trg.to(device)


def translate(forward_model, backward_model, seq: str, top: int) -> List[str]:
    """Round trip translation to generate top hypotheses"""
    forward_in_bin = forward_model.binarize(forward_model.apply_bpe(forward_model.tokenize(seq)))
    forward_out_bin = forward_model.generate(forward_in_bin, beam=1, verbose=True, sampling=True, sampling_topk=20)
    forward_translation = forward_model.decode(forward_out_bin[0]['tokens'])
    backward_in_bin = backward_model.binarize(backward_model.apply_bpe(backward_model.tokenize(forward_translation)))
    backward_out_bin = backward_model.generate(backward_in_bin, beam=top, verbose=True, sampling=True, sampling_topk=20)
    backward_translations_ids = [x['tokens'] for x in backward_out_bin]
    backward_translations = [backward_model.decode(x) for x in backward_translations_ids]
    return backward_translations


def translate_all(forward_model, backward_model, seqs: List[str], top: int):
    """Round trip translation for multiples sentences"""
    for seq in tqdm(seqs, desc='round trip translation', unit=' sentences'):
        trs = translate(forward_model, backward_model, seq, top)
        log_str = f'\n🔥 {seq}\n'
        for idx, tr in enumerate(trs):
            log_str += f'🔥 {idx + 1:>4d}/{top:<4d} {tr}\n'
        logger.info(log_str)


def load_aligner_model(device):
    """Load word alignment model"""
    model = simalign.SentenceAligner(
        model="bert",
        token_type="word",
        matching_methods="mai",
        device=device.type,
    )
    return model


def align(model, src_tokens: List[str], trg_tokens: List[str]):
    """Align source tokens and target tokens"""
    result = model.get_word_aligns(src_tokens, trg_tokens)
    alignment = result['mwmf']
    return [(src_tokens[i], trg_tokens[j]) for i, j in alignment]


def tokenize(seqs: List[str]):
    """Tokenize sequences"""
    docs = NLP.pipe(seqs, disable=['tagger', 'parser', 'ner'])
    tokenized_seqs = [[tok.text for tok in doc] for doc in tqdm(docs, desc='tokenize', total=len(seqs), leave=False)]
    return tokenized_seqs
