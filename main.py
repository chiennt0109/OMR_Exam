import sys

sys.dont_write_bytecode = True

from models.database import bootstrap_application_db


def main() -> None:
    bootstrap_application_db()
    from gui.main_window import run

    run()


if __name__ == "__main__":
    main()
