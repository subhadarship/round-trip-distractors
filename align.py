import os

import torch
from tqdm import tqdm

from utils import load_aligner_model, get_align_args, align, tokenize, read_round_trip_translations


def main():
    # get args
    args = get_align_args()

    # read round trip translations
    text, translations = read_round_trip_translations(
        translations_file_path=args.translations_file_path,
        top=args.top
    )

    # tokenize
    tokenized_text = tokenize(text)
    tokenized_translations = []
    for trs in tqdm(translations, desc='tokenize translations'):
        tokenized_translations.append(tokenize(trs))

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load aligner model
    model = load_aligner_model(device)

    # align
    lines_to_write = ''
    for inp, trs in tqdm(
            zip(tokenized_text, tokenized_translations),
            total=len(tokenized_text),
            unit=' seq'
    ):
        lines_to_write += 89 * '-' + '\n'
        for tr in tqdm(trs, desc='align', leave=False):
            aligned_pairs = align(model, inp, tr)
            lines_to_write += str(aligned_pairs) + '\n'
        lines_to_write += 89 * '-' + '\n'

    # write
    if os.path.split(args.out_file_path)[0] != '':
        os.makedirs(os.path.split(args.out_file_path)[0], exist_ok=True)
    with open(args.out_file_path, 'w', encoding='utf-8') as f:
        f.write(lines_to_write)


if __name__ == "__main__":
    main()
