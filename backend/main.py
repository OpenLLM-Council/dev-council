import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.cli import cli_main

if __name__ == "__main__":
    cli_main()
