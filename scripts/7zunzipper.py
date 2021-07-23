from pyunpack import Archive
import argparse
import os

parser = argparse.ArgumentParser(description="Unzips 7z file")
parser.add_argument(
    "input_path",
    action="store",
    help="input path to 7z file (including filename) to uncompress",
    type=str,
)
parser.add_argument(
    "-op",
    dest="output_path",
    action="store",
    default=None,
    help=(
        "output path to uncompress file (including filename), "
        "if not provided, uncompressed file is saved in same directory as of compressed file"
    ),
    type=None,
)
args = parser.parse_args()

ip = args.input_path
op = args.output_path

if not os.path.isfile(ip):
    raise ValueError(f"Expected path (including filename) to a 7z file")

if op is None:
    op = os.path.basename(ip).split(".")[0]
    op = os.path.join(os.path.dirname(ip), op)

os.makedirs(op, exist_ok=True)
Archive(ip).extractall(op)
print(ip, op)
