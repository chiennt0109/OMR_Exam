import sys

sys.dont_write_bytecode = True

def main() -> None:
    from gui.main_window import run

    run()


if __name__ == "__main__":
    main()
