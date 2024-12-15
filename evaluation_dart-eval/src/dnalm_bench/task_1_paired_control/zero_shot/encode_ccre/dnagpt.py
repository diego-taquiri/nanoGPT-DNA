import os
import sys
from pathlib import Path
import torch

# Add the parent directory to Python path to import train_gpt_dna
sys.path.append(str(Path(__file__).parents[6]))
from train_gpt_dna import GPT, GPTConfig

from ..evaluators import PairedControlDataset, DNAGPTEvaluator

os.environ["TOKENIZERS_PARALLELISM"] = "false"

work_dir = os.environ.get("DART_WORK_DIR", "")

if __name__ == "__main__":
    model_name = "DNA-GPT"

    genome_fa = os.path.join(work_dir, "refs/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta")
    elements_tsv = os.path.join(work_dir, f"task_1_ccre/processed_inputs/ENCFF420VPZ_processed.tsv")

    out_dir = os.path.join(work_dir, f"task_1_ccre/zero_shot_outputs/likelihoods/{model_name}")

    chroms = ["chr21"]  # For testing

    batch_size = 32  # Reduced from 256
    num_workers = 4
    seed = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    max_elements = 128  # Limit for testing
    model_path = "log/model.pt" 
    dataset = PairedControlDataset(genome_fa, elements_tsv, chroms, seed, max_elements=max_elements)
    evaluator = DNAGPTEvaluator(model_path, dataset, batch_size, num_workers, device)
    metrics = evaluator.evaluate(out_dir, progress_bar=True)

    for k, v in metrics.items():
        print(f"{k}: {v}") 