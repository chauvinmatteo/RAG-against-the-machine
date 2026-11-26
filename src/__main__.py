from src.cli import RagCLI
from fire import Fire


def main() -> None:

    Fire(RagCLI)


if __name__ == "__main__":
    main()
