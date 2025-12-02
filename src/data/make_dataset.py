import argparse
from pathlib import Path
from .simulate_behaviors import generate_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=2000)
    parser.add_argument("--length", type=int, default=24 * 14)
    parser.add_argument("--fake_ratio", type=float, default=0.35)
    parser.add_argument("--output_path", type=str, default="data/raw/engagement.parquet")
    args = parser.parse_args()

    df = generate_dataset(
        n_users=args.n_users,
        length=args.length,
        fake_ratio=args.fake_ratio,
    )

    output = Path(args.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output, index=False)


if __name__ == "__main__":
    main()

