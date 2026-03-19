import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import torchvision.transforms as transforms

# --- 1. THE VOCABULARY ---
class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.itos = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4 

        for sentence in sentence_list:
            for word in str(sentence).lower().split():
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = str(text).lower().split()
        return [self.stoi["<START>"]] + \
               [self.stoi.get(word, self.stoi["<UNK>"]) for word in tokenized_text] + \
               [self.stoi["<END>"]]

# --- 2. THE DATASET ---
class FlickrDataset(Dataset):
    def __init__(self, image_dir, captions_file, transform=None, freq_threshold=5):
        self.image_dir = image_dir
        self.transform = transform
        
        # Load the captions file using Pandas
        self.df = pd.read_csv(captions_file)
        
        # Extract image names and captions
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        # Initialize and build the vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        
        # Load Image
        img_path = os.path.join(self.image_dir, img_id)
        img = Image.open(img_path).convert("RGB")
        
        # Apply transformations (Resize and convert to Tensor)
        if self.transform is not None:
            img = self.transform(img)
            
        # Convert text to numbers
        numericalized_caption = [self.vocab.stoi["<START>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        
        return img, torch.tensor(numericalized_caption)

# --- 3. THE COLLATER (For Padding Sentences) ---
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Separate images and captions from the batch
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        captions = [item[1] for item in batch]
        
        # Pad the captions so they are all the same length in this batch
        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)
        
        return imgs, captions

# --- 4. DATA LOADER HELPER FUNCTION ---
def get_loader(image_dir, captions_file, transform, batch_size=32, num_workers=0, shuffle=True):
    dataset = FlickrDataset(image_dir, captions_file, transform=transform)
    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )
    return loader, dataset

# --- 5. TEST IT ON YOUR REAL DATA ---
if __name__ == "__main__":
    print("Loading real Flickr8k data...")
    
    # Path to your extracted folders
    IMG_DIR = "flickr8k/Images"
    CAPTIONS_FILE = "flickr8k/captions.txt"
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Create the loader (Batch size of 4 images at a time)
    dataloader, dataset = get_loader(IMG_DIR, CAPTIONS_FILE, transform=transform, batch_size=4)
    
    print(f"Total vocabulary size: {len(dataset.vocab)} unique words.")
    print("Fetching one batch of data...")
    
    # Grab one batch of data to test
    iterator = iter(dataloader)
    imgs, captions = next(iterator)
    
    print(f"\nSUCCESS!")
    print(f"Images batch shape: {imgs.shape} (4 images, 3 color channels, 256x256 pixels)")
    print(f"Captions batch shape: {captions.shape} (4 sentences, padded to the longest one)")