"""Run final inference from the locked development selection (see experiment.py)."""
from experiment import main

if __name__ == "__main__":
    import sys
    if "--mode" not in sys.argv:
        sys.argv[1:1] = ["--mode", "final"]
    main()
