"""OOD score comparison for fashion_mnist; see common.py for the protocol."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments_ood.common import main

if __name__ == "__main__":
    main("fashion_mnist")
