import torch
import os
import numpy as np
from torch.utils.data import Dataset


class VideoDataset(Dataset):
    """
    Word-level GRID dataset using speaker-based filtering.

    Structure:
    D:/grid_word_dataset/
        bin/
            s1_xxxx.npy
        lay/
        place/
        set/
    """

    def __init__(self, root_dir, speakers):
        self.root_dir = root_dir
        self.label_list = ['bin', 'lay', 'place', 'set']
        self.speakers = speakers
        self.pathList = self.build_list()

    def build_list(self):
        list_of_data = []

        for label_idx, label in enumerate(self.label_list):
            label_dir = os.path.join(self.root_dir, label)

            if not os.path.exists(label_dir):
                continue

            for file in os.listdir(label_dir):

                if file.endswith(".npy"):

                    # Extract speaker ID (example: s18_xxx.npy)
                    speaker_id = file.split("_")[0]

                    if speaker_id in self.speakers:
                        file_path = os.path.join(label_dir, file)
                        list_of_data.append((label_idx, file_path))

        return list_of_data

    def __len__(self):
        return len(self.pathList)

    def __getitem__(self, idx):

        label, path = self.pathList[idx]

        video_np = np.load(path)  # shape: (T, H, W, C)

        # ----------------------------------
        # FORCE FIXED LENGTH (29 FRAMES)
        # ----------------------------------
        MAX_FRAMES = 29
        T = video_np.shape[0]

        if T > MAX_FRAMES:
            video_np = video_np[:MAX_FRAMES]

        elif T < MAX_FRAMES:
            pad = MAX_FRAMES - T
            last_frame = video_np[-1]
            pad_frames = np.repeat(last_frame[np.newaxis, ...], pad, axis=0)
            video_np = np.concatenate((video_np, pad_frames), axis=0)

        # ----------------------------------
        # Normalize
        # ----------------------------------
        video_np = video_np.astype(np.float32) / 255.0

        # Convert to tensor
        video_tensor = torch.from_numpy(video_np)
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # (C, T, H, W)

        return video_tensor, torch.tensor(label, dtype=torch.long)
