"""Run one migrated deep CT prior on a source toy data pack."""
import argparse
import json
import numpy as np
from inv_framework.ct.toy import toy_experiment
from inv_framework.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--algorithm', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--weights')
    parser.add_argument('--device')
    parser.add_argument('--params', default='{}', help='JSON parameter overrides')
    args = parser.parse_args()
    params = json.loads(args.params)
    with np.load(args.input, allow_pickle=False) as data:
        for key in ('image_scale', 'image_offset'):
            if key in data:
                params.setdefault(key, float(data[key]))
    if args.weights:
        params['weights'] = args.weights
    config = toy_experiment(args.input, args.algorithm, device=args.device, params=params)
    result = run_experiment(config, output_dir=args.output)
    print(result.status, result.stopping_reason)
    if result.status != 'success':
        raise SystemExit(str(result.error))


if __name__ == '__main__':
    main()
