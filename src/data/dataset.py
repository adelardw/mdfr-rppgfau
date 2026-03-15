import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

class RecursiveFolderDataset(Dataset):
    """
    for ff++ and celebDF
    """
    def __init__(self, root_dir, transform=None, valid_extensions=('.jpg', '.jpeg', '.png', '.bmp')):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Folder {root_dir} not found")

        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        for idx, class_name in enumerate(subdirs):
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx

            class_folder = os.path.join(root_dir, class_name)

            for root, _, files in os.walk(class_folder):
                for file in files:
                    if file.lower().endswith(valid_extensions):
                        path = os.path.join(root, file)
                        self.samples.append((path, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Ошибка при загрузке {path}: {e}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, label



class VideoFolderDataset_OLD(Dataset):
    """
    Dataset for ff++ and celebDF
    Reads video files and returns a sequence of frames.
    """
    def __init__(self, root_dir, transform=None, valid_extensions=('.mp4', '.avi', '.mov'), frames_per_video=16):
        self.root_dir = root_dir
        self.transform = transform
        self.valid_extensions = valid_extensions
        self.frames_per_video = frames_per_video

        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Folder {root_dir} not found")

        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

        for idx, class_name in enumerate(subdirs):
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx

            class_folder = os.path.join(root_dir, class_name)

            for root, _, files in os.walk(class_folder):
                for file in files:
                    if file.lower().endswith(self.valid_extensions):
                        path = os.path.join(root, file)
                        self.samples.append((path, idx))

    def __len__(self):
        return len(self.samples)

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []

        if not cap.isOpened():
            print(f"ERROR : {path}")
            return self._get_dummy_video()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return self._get_dummy_video()

        frame_indices = np.linspace(0, total_frames - 1, self.frames_per_video, dtype=int)

        current_frame_idx = 0
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            else:
                print("attention!. See class dataset")
                frames.append(Image.new('RGB', (224, 224))) # todo: del

        cap.release()
        while len(frames) < self.frames_per_video:
            frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))

        return frames

    def _get_dummy_video(self):
        return [Image.new('RGB', (224, 224)) for _ in range(self.frames_per_video)]

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            frames = self._load_video(path)
        except Exception as e:
            print(f"Ошибка при загрузке {path}: {e}")
            frames = self._get_dummy_video()

        if self.transform:
            frames = [self.transform(img) for img in frames]

        video_tensor = torch.stack(frames)

        return video_tensor.transpose(0,1), label



class VideoFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None, video_transform=None,
                 valid_extensions=('.mp4', '.avi', '.mov', '.mkv'), frames_per_video=32):
        self.root_dir = root_dir
        self.transform = transform
        self.video_transform = video_transform
        self.valid_extensions = valid_extensions
        self.frames_per_video = frames_per_video

        self.samples = []
        self.classes = []
        self.class_to_idx = {}

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Folder {root_dir} not found")
        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        for idx, class_name in enumerate(subdirs):
            self.classes.append(class_name)
            self.class_to_idx[class_name] = idx
            class_folder = os.path.join(root_dir, class_name)

            for root, _, files in os.walk(class_folder):
                for file in files:
                    if file.lower().endswith(self.valid_extensions):
                        path = os.path.join(root, file)
                        self.samples.append((path, idx))

        print(f"Found {len(self.samples)} videos in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def _get_dummy_video(self):
        return [Image.new('RGB', (224, 224)) for _ in range(self.frames_per_video)]

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        
        if not cap.isOpened():
            print(f"Error opening {path}")
            return self._get_dummy_video()

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return self._get_dummy_video()

        clip_len = self.frames_per_video
        
        if total_frames > clip_len:
            start_frame = np.random.randint(0, total_frames - clip_len)
        else:
            start_frame = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frames = []
        for _ in range(clip_len):
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            else:
                break
        cap.release()

        if len(frames) == 0: return self._get_dummy_video()

        original_len = len(frames)
        while len(frames) < clip_len:
            frames.append(frames[len(frames) % original_len])
            
        return frames[:clip_len]

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            frames = self._load_video(path)
        except Exception as e:
            frames = self._get_dummy_video()

        if self.video_transform:
            video_tensor = self.video_transform(frames)
        elif self.transform:
            frames = [self.transform(img) for img in frames]
            video_tensor = torch.stack(frames).permute(1, 0, 2, 3)
        else:
            raise ValueError("No transform or video_transform provided")

        return video_tensor, label



def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Need sum == 1"

    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    return train_set, val_set, test_set

if __name__ == "__main__":

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


    #full_dataset = RecursiveFolderDataset(root_path, transform=data_transforms)

    # print(f"Classes: {full_dataset.classes}")
    # print(f"Len images: {len(full_dataset)}")
    # train_ds, val_ds, test_ds = split_dataset(full_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    # train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    # val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    # test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # images, labels = next(iter(train_loader))
    # print(f": {images.shape}")
    # print(f"Labels: {labels}")


    root_path = "../datasets/ff++_videous_out"


    full_dataset = VideoFolderDataset(root_path, transform=data_transforms)

    print(f"Classes: {full_dataset.classes}")
    print(f"Len images: {len(full_dataset)}")
    train_ds, val_ds, test_ds = split_dataset(full_dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    images, labels = next(iter(train_loader))
    print(f": {images.shape}")
    print(f"Labels: {labels}")