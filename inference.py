"""Given a path to a model checkpoint and a dataset split ('test', 'train', 'validation', 'all'),
compute all metrics.
"""
import argparse
import os
import sys
import torch
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.roformer import Roformer
from tokenizer import MultistreamTokenizer
from utils import eval, infer, pad_batch
from lightning.pytorch import seed_everything
from score_utils import postprocess_score

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_file", type=str, default="../../piano-cover-generation-main/base_model/covers/test.mid")
    parser.add_argument("--output_file", type=str, default="output.xml")
    parser.add_argument("--model", type=str, default="../MIDI2ScoreTF.ckpt")

    args = parser.parse_args()

    seed_everything(42, workers=True)

    print("Loading model")
    model = Roformer.load_from_checkpoint(args.model)
    model.to(device)
    model.eval()

    print("Running inference")
    x = MultistreamTokenizer.tokenize_midi(args.midi_file)
    y_hat = infer(x, model)
    mxl = MultistreamTokenizer.detokenize_mxl(y_hat)
    mxl = postprocess_score(mxl)
  
    print(f"Writing sheet music to {args.output_file}")
    mxl.write("musicxml",fp=args.output_file)
