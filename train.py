import argparse
import json
import torch
from core.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--c', type=str, required=True, help='Path to JSON config')
    args = parser.parse_args()

    with open(args.c) as f:
        config = json.load(f)

    config['device']      = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['distributed'] = False
    config['world_size']  = 1
    config['global_rank'] = 0
    config['local_rank']  = 0

    print(f'Device: {config["device"]}')
    print(f'Config: {args.c}')

    trainer = Trainer(config)
    trainer.train()
