from pathlib import Path
from src.data_loader import load_dataset
from src.candidate_routes import get_candidates

root = Path(__file__).resolve().parent

# Select dataset
data = load_dataset(root / "dataset/mandl/basis")
# data = load_dataset(root / "dataset/toy/basis")

# Candidate generation parameters
k = 3
max_time = 30
remove_subpaths = True

candidates = get_candidates(data, k=k, max_time=max_time, 
                            remove_subpaths=remove_subpaths)

print("Candidate routes:", len(candidates))