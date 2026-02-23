import sys
import os
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import run


def parse_args():
    parser = argparse.ArgumentParser(description="Dev Council CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("onboard", help="Onboard Ollama models")

    subparsers.add_parser("code", help="Invoke the main dev-council workflow")

    return parser.parse_args()


def cli_main():
    if len(sys.argv) == 1:
        run()
    else:
        args = parse_args()
        if args.command == "onboard":
            from app.onboard import onboard_ollama

            onboard_ollama()
        elif args.command == "code":
            run()
        else:
            print("Unknown command or missing sub-command. Use -h for help.")
            sys.exit(1)


if __name__ == "__main__":
    cli_main()
