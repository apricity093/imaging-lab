"""Run source toy CT data through the public experiment contract."""
import argparse
import json
from inv_framework.ct.toy import toy_experiment
from inv_framework.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--algorithm', default='sirt')
    parser.add_argument('--device')
    parser.add_argument('--params', type=json.loads, default={})
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = toy_experiment(args.input, args.algorithm, device=args.device, params=args.params)
    result = run_experiment(config, output_dir=args.output)
    print(json.dumps(dict(status=result.status, parameters=result.effective_params,
                         metrics=result.metrics, error=result.error), ensure_ascii=False))
    return 0 if result.status == 'success' else 1


if __name__ == '__main__':
    raise SystemExit(main())
