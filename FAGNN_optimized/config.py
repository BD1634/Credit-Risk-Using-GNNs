#config.py
import torch
import argparse
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--data_load_path', default='./models/')
parser.add_argument('--theta_m', type=float, default=0.3)
parser.add_argument('--theta_u', type=int, default=20)
parser.add_argument('--up_sample', type=float, default=0.0)
parser.add_argument('--down_sample', type=float, default=0.0)
parser.add_argument('--valid_portion', type=float, default=0.1)
parser.add_argument('--optimizer', default=optim.Adam)
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--step_size', type=float, default=50)
parser.add_argument('--EmbeddingSize', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--l2', type=float, default=0)
parser.add_argument('--batchSize', type=int, default=500)
parser.add_argument('--lambda_', type=float, default=0.5)
parser.add_argument('--alpha_', type=float, default=0.5)
parser.add_argument('--beta_', type=float, default=0.0)
opt, _ = parser.parse_known_args()

def get_device():
    if torch.cuda.is_available():
        print("Using: CUDA")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using: MPS")
        return torch.device("mps")
    print("Using: CPU")
    return torch.device("cpu")