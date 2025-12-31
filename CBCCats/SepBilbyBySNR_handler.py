import argparse
import SepBilbyBySNR


def main(run):
    for r in run:
        SepBilbyBySNR.main(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r", "--run", type=int, help="The run number for the specific subset"
    )
    args = parser.parse_args()
    main(args.run)
