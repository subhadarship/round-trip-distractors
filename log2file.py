import os

from utils import read_log, get_log2file_args


def main():
    # get args
    args = get_log2file_args()

    # read lines
    lines = read_log(args.log_file_path)

    # write lines
    if os.path.split(args.write_file_path)[0] != '':
        os.makedirs(os.path.split(args.write_file_path)[0], exist_ok=True)
    with open(args.write_file_path, 'w', encoding='utf-8') as f:
        f.writelines(list(map(lambda s: s + '\n', lines)))


if __name__ == "__main__":
    main()
