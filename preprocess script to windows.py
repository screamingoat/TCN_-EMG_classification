import os
import numpy as np
from biosppy.signals import emg
from concurrent.futures import ThreadPoolExecutor
import tqdm
import h5py 

# Config
WIN   = 512
STEP  = WIN - 128
FS    = 2048
GESTURE_LIST = list(range(17))
N_CLASSES = len(GESTURE_LIST)
GESTURE_MAP = {g: i for i, g in enumerate(GESTURE_LIST)}

SESSION_ROOTS = [
    "/home/monsef/Desktop/Output BM/Session1_converted",
    "/home/monsef/Desktop/Output BM/Session2_converted",
    "/home/monsef/Desktop/Output BM/Session3_converted"
]

TRAIN_SUB = range(1, 35)
TEST_SUB  = range(35, 43)

CHUNK_SIZE = 5000  # windows per saved chunk

def apply_biosppy_filters(signal, fs=FS):
    def filter_ch(ch):
        out = emg.emg(signal=ch, sampling_rate=fs, show=False)
        return out['filtered']
    with ThreadPoolExecutor() as pool:
        filtered = list(pool.map(filter_ch, signal.T))
    filtered = np.array(filtered).T.astype(np.float32)
    mean = np.mean(filtered, axis=0, keepdims=True)
    std = np.std(filtered, axis=0, keepdims=True) + 1e-8
    normed = (filtered - mean) / std
    return normed

def get_pair(session, pid):
    base = f"session{session}_participant{pid}"
    fore = os.path.join(SESSION_ROOTS[session-1], base + "_DATA_FOREARM.npy")
    wrst = os.path.join(SESSION_ROOTS[session-1], base + "_DATA_WRIST.npy")
    return fore, wrst

def collect_and_save_chunks(subjects, label_map, prefix="train"):
    fore_idx = np.array(list(range(0, 16)) + list(range(17, 23)) + list(range(25, 28)))
    wrst_idx = np.array([0, 1, 2])
    chunk_count = 0
    windows = []
    labels = []
    for ses in range(1, 4):
        for pid in tqdm.tqdm(subjects, desc=f"{prefix} Session {ses}"):
            fore_path, wrst_path = get_pair(ses, pid)
            if not (os.path.isfile(fore_path) and os.path.isfile(wrst_path)):
                continue
            fore = np.load(fore_path, allow_pickle=True)
            wrst = np.load(wrst_path, allow_pickle=True)
            for trial in range(7):
                for g in GESTURE_LIST:
                    f_seg = fore[trial, g]
                    w_seg = wrst[trial, g]
                    if f_seg is None or w_seg is None:
                        continue
                    f_sel = f_seg[:, fore_idx]
                    w_sel = w_seg[:, wrst_idx]
                    seg = np.concatenate([f_sel, w_sel], axis=1).astype(np.float32)
                    seg = apply_biosppy_filters(seg, fs=FS)
                    for start in range(0, seg.shape[0] - WIN + 1, STEP):
                        win = seg[start:start + WIN]
                        windows.append(win)
                        labels.append(label_map[g])
                        # Save chunk if chunk_size reached
                        if len(windows) == CHUNK_SIZE:
                            np.save(f"{prefix}_windows_{chunk_count}.npy", np.stack(windows))
                            np.save(f"{prefix}_labels_{chunk_count}.npy", np.array(labels))
                            print(f"Saved chunk {chunk_count} ({len(windows)} windows)")
                            chunk_count += 1
                            windows = []
                            labels = []
    # Save last chunk
    if windows:
        np.save(f"{prefix}_windows_{chunk_count}.npy", np.stack(windows))
        np.save(f"{prefix}_labels_{chunk_count}.npy", np.array(labels))
        print(f"Saved chunk {chunk_count} ({len(windows)} windows)")

if __name__ == "__main__":
    collect_and_save_chunks(TRAIN_SUB, GESTURE_MAP, prefix="train")
    collect_and_save_chunks(TEST_SUB,  GESTURE_MAP, prefix="test")
