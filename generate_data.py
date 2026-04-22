import numpy as np
import os
from tqdm import tqdm
from rubik_env import RubikCube, ALL_MOVES

def sample_depth():
    bins = [1, 4, 8, 13, 16, 21]
    probs = [0.10, 0.30, 0.40, 0.15, 0.05]
    interval = np.random.choice(len(probs), p=probs)
    return np.random.randint(bins[interval], bins[interval+1])

def generate_dataset(num_samples=2_000_000, save_path='data/dataset_2M.npz'):
    cube = RubikCube()
    solved_state = cube.get_solved_state().copy()
    
    X = np.zeros((num_samples, 54), dtype=np.uint8)
    y = np.zeros((num_samples,), dtype=np.float32)
    
    for i in tqdm(range(num_samples), desc="Generating data"):
        k = sample_depth()
        cube.set_state(solved_state.copy())
        for _ in range(k):
            move = np.random.choice(ALL_MOVES)
            cube.apply_move(move)
        X[i] = cube.get_state()
        y[i] = k
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, X=X, y=y)

if __name__ == "__main__":
    generate_dataset(num_samples=2_000_000, save_path='data/dataset_2M.npz')
    
