from augment_utils import *
import argparse

parser = argparse.ArgumentParser(
    description='Create literals and append to graph'
)
parser.add_argument('--dataset', default='YAGO3-10', metavar='',
                    help='which dataset in {`YAGO3-10`, `FB15k`, `FB15k-237`} to be used? (default: YAGO3-10)')
parser.add_argument('--mode', default='All', metavar='',
                    help='which augmentation mode to run?')
parser.add_argument('--bins', default='8', metavar='',
                    help='How many bins to run?')
parser.add_argument('--levels', default='3', metavar='',
                    help='How many levels to run?')
args = parser.parse_args()

### Read configuration
home = os.environ['HOME']
dataset = args.dataset
modes = args.mode.split(',')
bins = int(args.bins)
levels = int(args.levels)

if modes[0] == "All":
    modes = SUPPORTED_MODE

if dataset == "WikidataDWD":
    try:
        modes.remove("Jenks")
    except:
        pass

# Get data
entities, values = get_data_np(dataset)

for mode in modes:
    if mode in SUPPORTED_MODE:
        augment_np(entities, values, dataset, mode, bins, levels)
