import argparse
import json
parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='4SQ_TKY')
parser.add_argument('--eps', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--img_size', type=int, default=128)
parser.add_argument('--comment', type=str, default='')
parser.add_argument('--cell_size', type=int, default=576)
parser.add_argument('--window_size', type=int, default=8)
parser.add_argument('--drop_rate', type=float, default=0.0)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--model', type=str, default='mv')
parser.add_argument('--test', action='store_true')
parser.add_argument('--patch_size', type=int, default=1)
parser.add_argument('--data_norm', action='store_true')
parser.add_argument('--emb_dim', type=int, default=180)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--file_config', action='store_true')
parser.add_argument('--depth', type=str, default='[6, 6, 6, 6]')
parser.add_argument('--scheme', type=str, default='c1')
parser.add_argument('--truncate_size', type=str, default=5)
# read from json
config = json.load(open('./config.json'))

args = parser.parse_args()
if not args.file_config:
    config['datasets']['name'] = args.dataset
    config['privacy']['eps'] = args.eps
    config['privacy']['scheme'] = args.scheme
    config['train']['lr'] = args.lr
    config['train']['gpu'] = args.gpu
    config['net']['img_size'] = args.img_size
    config['datasets']['cell_size'] = args.cell_size
    config['datasets']['truncate_size'] = args.truncate_size
    config['train']['comment'] = args.comment
    config['net']['window_size'] = args.window_size
    config['net']['drop_rate'] = args.drop_rate
    config['train']['epochs'] = args.epochs
    config['train']['model'] = args.model
    config['is_train'] = not args.test
    config['datasets']['patch_size'] = args.patch_size
    config['train']['data_norm'] = args.data_norm
    config['net']['embed_dim'] = args.emb_dim
    config['train']['alpha'] = args.alpha
    config['net']['depth'] = eval(args.depth)
