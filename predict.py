import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
from model import Encoder, DecoderWithAttention
from dataset import get_loader

def plot_attention_heatmap(image_path, caption_words, alphas):
    """
    Takes the generated words and attention weights and plots a grid of heatmaps.
    """
    print("Generating attention heatmaps for your PPT...")
    img = Image.open(image_path).convert("RGB")
    
    # Create a big plot figure
    fig = plt.figure(figsize=(15, 15))
    num_words = len(caption_words)
    
    for i in range(num_words):
        # Create a grid layout (4 images per row)
        ax = fig.add_subplot(int(np.ceil(num_words / 4.0)), 4, i + 1)
        
        # Show the original image
        ax.imshow(img)
        
        # Reshape the 1D alpha tensor into a 2D square (e.g., 64 -> 8x8)
        alpha_tensor = alphas[i].squeeze()
        grid_size = int(math.sqrt(alpha_tensor.shape[0]))
        alpha_2d = alpha_tensor.view(grid_size, grid_size).numpy()
        
        # Resize the heatmap to match the original image size
        alpha_img = Image.fromarray(alpha_2d)
        alpha_img = alpha_img.resize(img.size, Image.Resampling.LANCZOS)
        
        # Overlay the heatmap (cmap='jet' makes it red/blue/yellow)
        ax.imshow(np.array(alpha_img), alpha=0.6, cmap='jet')
        
        # Put the word as the title of this specific heatmap
        ax.set_title(caption_words[i], fontsize=16)
        ax.axis('off')
        
    plt.tight_layout()
    plt.show()

def generate_caption(image_path, encoder, decoder, vocab, max_length=20):
    print(f"Reading image: {image_path}...")
    
    # 1. Prepare the image exactly how the AI expects it
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0) # Add a fake batch dimension of 1
    
    # 2. Put models in "Evaluation" mode
    encoder.eval()
    decoder.eval()
    
    caption_words = []
    saved_alphas = [] # <-- NEW: List to hold our heatmaps!
    
    with torch.no_grad():
        # 3. Extract image features ("Show")
        features = encoder(image_tensor)
        
        # 4. Initialize the LSTM's hidden state
        h, c = decoder.init_hidden_state(features)
        
        # 5. Start the sentence
        word = torch.tensor([vocab.stoi["<START>"]])
        
        # 6. Generate words one by one ("Attend and Tell")
        for i in range(max_length):
            embeddings = decoder.embedding(word)
            
            # Calculate where the AI should look (Attention)
            attention_weighted_encoding, alpha = decoder.attention(features, h)
            
            # <-- NEW: Save the attention math to our list
            saved_alphas.append(alpha.detach().cpu())
            
            # Combine the word and the visual attention, pass to LSTM
            lstm_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
            h, c = decoder.decode_step(lstm_input, (h, c))
            
            # Predict the next word
            output = decoder.fc(h)
            predicted_word_idx = output.argmax(1).item()
            predicted_word = vocab.itos[predicted_word_idx]
            
            # Stop if the AI says the sentence is over
            if predicted_word == "<END>":
                break
                
            caption_words.append(predicted_word)
            
            # Feed this predicted word into the next loop iteration
            word = torch.tensor([predicted_word_idx])
            
    # Return both the list of words AND the list of heatmaps
    return caption_words, saved_alphas

if __name__ == "__main__":
    print("Setting up Inference environment...")
    
    IMG_DIR = "flickr8k/Images"
    CAPTIONS_FILE = "flickr8k/captions.txt"
    
    _, dataset = get_loader(IMG_DIR, CAPTIONS_FILE, transform=None, batch_size=1)
    vocab = dataset.vocab
    
    encoder = Encoder()
    decoder = DecoderWithAttention(attention_dim=256, embed_dim=256, decoder_dim=256, vocab_size=len(vocab))
    
    try:
        print("Loading trained weights...")
        device = torch.device('cpu')
        # Pointing to the 14-epoch files you are training right now!
        encoder.load_state_dict(torch.load('encoder_epoch_14.pth', map_location=device, weights_only=True))
        decoder.load_state_dict(torch.load('decoder_epoch_14.pth', map_location=device, weights_only=True))
        print("Weights loaded successfully!")
    except FileNotFoundError:
        print("⚠️ WARNING: Could not find your .pth files.")
        print("⚠️ Make sure you downloaded them from Colab and put them in this folder!")
    
    # Pick a test image 
    test_image_path = "p10.jpg" 
    
    # Generate the caption and the heatmaps!
    words, attention_maps = generate_caption(test_image_path, encoder, decoder, vocab)
    
    if words[0] == "<START>":
        words = words[1:]
        attention_maps = attention_maps[1:]
        
    # Just in case <END> somehow sneaks into the very back, chop that off too
    if words[-1] == "<END>":
        words = words[:-1]
        attention_maps = attention_maps[:-1]
    
    # Join the words together for the final text output
    final_sentence = " ".join(words)
    print("\n=========================================")
    print(f"AI'S CAPTION: {final_sentence}")
    print("=========================================\n")
    
    # Pop up the visual grid!
    plot_attention_heatmap(test_image_path, words, attention_maps)