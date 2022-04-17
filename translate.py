import logging
import os

import torch

logger = logging.getLogger(__name__)
from utils import get_translate_args, init_logger, load_mt_model, translate_all


def main():
    # get args
    args = get_translate_args()
    # initialize logger
    init_logger(args.log_file_path)

    # sanity check
    assert os.path.isfile(args.src_file_path)

    with open(args.src_file_path, 'r', encoding='utf-8') as f:
        sentences = list(map(str.strip, f.readlines()))

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load models
    f_model = load_mt_model(src_lang=args.src_lang, trg_lang=args.pivot_lang, device=device)
    b_model = load_mt_model(src_lang=args.pivot_lang, trg_lang=args.src_lang, device=device)

    # round trip translation
    translate_all(forward_model=f_model, backward_model=b_model, seqs=sentences, top=args.top)

    logger.info('COMPLETED')


if __name__ == "__main__":
    main()
