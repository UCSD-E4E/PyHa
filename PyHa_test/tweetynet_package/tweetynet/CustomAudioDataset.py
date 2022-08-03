from torch.utils.data import Dataset

class CustomAudioDataset(Dataset):
    def __init__(self, spec, annotations, uids, transform=None, target_transform=None):
        self.img_labels = annotations
        self.spec = spec
        self.uids = uids
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.img_labels[idx, 0])
        #Read_audio
        image = self.spec[idx]#read_image(img_path)
        label = self.img_labels[idx]
        # transform should be spectrogram from librosa # unless we do that beforehand.
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, self.uids[idx]

