import argparse

from sentiment_analysis.config import FINAL_MODEL_PATH, LABEL_NAMES
from sentiment_analysis.persistence import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict sentiment for a tweet using the trained model."
    )
    parser.add_argument(
        "text",
        type=str,
        help="Tweet text to classify.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = load_model(FINAL_MODEL_PATH)

    prediction = pipeline.predict([args.text])[0]
    label = LABEL_NAMES[int(prediction)]

    print(label)


if __name__ == "__main__":
    main()
