import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["mic1", "mic2"], default="mic2")
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    args = parser.parse_args()

    # remove 's5', 'p280', 'p315'
    for spk in os.listdir(args.input_dir):
        if not os.path.isdir(os.path.join(args.input_dir, spk)):
            continue
        if spk in ('s5', 'p280', 'p315'):
            continue
        spk_dir = os.path.join(args.output_dir, spk)
        os.makedirs(spk_dir, exist_ok=True)
        for filename in os.listdir(os.path.join(args.input_dir, spk)):
            if not filename.endswith(f"{args.type}.flac"):
                continue
            input_file = os.path.join(args.input_dir, spk, filename)
            output_file = os.path.join(args.output_dir, spk, filename)
            os.system(f"ln -s {input_file} {output_file}")
                

if __name__ == '__main__':
    main()
