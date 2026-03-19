import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import os # <-- Added for creating safe folders

from dataset import get_loader
from model import Encoder, DecoderWithAttention

def train():
    print("Setting up training environment...")
    
    # --- 1. HYPERPARAMETERS ---
    embed_dim = 256
    attention_dim = 256
    decoder_dim = 256
    batch_size = 32
    learning_rate = 3e-4
    num_epochs = 30 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 2. LOAD THE DATA ---
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    IMG_DIR = "flickr8k/Images"
    CAPTIONS_FILE = "flickr8k/captions.txt"
    
    print("Loading dataset (this might take a minute)...")
    train_loader, dataset = get_loader(
        image_dir=IMG_DIR, 
        captions_file=CAPTIONS_FILE, 
        transform=transform, 
        batch_size=batch_size
    )
    vocab_size = len(dataset.vocab)

    # --- 3. INITIALIZE THE MODEL ---
    print("Initializing the Show, Attend and Tell model...")
    encoder = Encoder().to(device)
    decoder = DecoderWithAttention(
        attention_dim=attention_dim,
        embed_dim=embed_dim,
        decoder_dim=decoder_dim,
        vocab_size=vocab_size
    ).to(device)

    # --- 4. LOSS AND OPTIMIZER ---
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(list(decoder.parameters()) + list(encoder.resnet.parameters()), lr=learning_rate)

    # --- 5. THE TRAINING LOOP ---
    print("Starting training! Let the AI learn...")
    
    for epoch in range(num_epochs):
        for idx, (imgs, captions) in enumerate(train_loader):
            
            imgs = imgs.to(device)
            captions = captions.to(device)

            features = encoder(imgs)
            predictions, alphas = decoder(features, captions, [len(c) for c in captions])

            targets = captions[:, 1:] 
            predictions = predictions.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = criterion(predictions, targets)

            optimizer.zero_grad() 
            loss.backward()       
            optimizer.step()

            if idx % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Batch [{idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")
                
        # --- BULLETPROOF SAVING ---
        # Create a safe folder in your actual Google Drive
        save_dir = '/content/drive/MyDrive/AI_Backups/'
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"--> Epoch {epoch+1} finished! Saving safely to Google Drive...")
        
        # Save directly to the Drive path!
        torch.save(encoder.state_dict(), f'{save_dir}encoder_epoch_{epoch+1}.pth')
        torch.save(decoder.state_dict(), f'{save_dir}decoder_epoch_{epoch+1}.pth')   
        
    print("Training Complete! The model is now much smarter.")
    
    # --- 6. SAVE THE FINAL MODEL ---
    print("Saving the AI's final brain safely to your Google Drive...")
    torch.save(encoder.state_dict(), f'{save_dir}encoder_weights_final.pth')
    torch.save(decoder.state_dict(), f'{save_dir}decoder_weights_final.pth')
    print("Saved successfully! You can now run predict.py")

if __name__ == "__main__":
    train()