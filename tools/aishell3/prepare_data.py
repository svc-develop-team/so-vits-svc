import argparse
import os


def prepare_data(input_dir, output_dir):
    for spk in os.listdir(input_dir):
        os.makedirs(os.path.join(output_dir, spk), exist_ok=True)
        for filename in os.listdir(os.path.join(input_dir, spk)):
            if not filename.endswith(".wav"):
                continue
            input_file = os.path.join(input_dir, spk, filename)
            output_file = os.path.join(output_dir, spk, filename)
            os.system(f"ln -s {input_file} {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    args = parser.parse_args()

    prepare_data(os.path.join(args.input_dir, "train/wav"), args.output_dir)
    prepare_data(os.path.join(args.input_dir, "test/wav"), args.output_dir)


if __name__ == '__main__':
    main()
