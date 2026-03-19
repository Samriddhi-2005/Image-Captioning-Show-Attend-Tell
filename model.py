import torch
import torch.nn as nn
import torchvision.models as models

# --- 1. THE ENCODER ("Show") ---
class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        # Load a pre-trained ResNet to act as our "eyes"
        resnet = models.resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        out = self.resnet(images) 
        out = self.adaptive_pool(out) 
        out = out.permute(0, 2, 3, 1)
        return out.view(out.size(0), -1, out.size(-1))

# --- 2. THE ATTENTION MECHANISM ("Attend") ---
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

# --- 3. THE DECODER ("Tell") ---
class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048):
        super(DecoderWithAttention, self).__init__()
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        return self.init_h(mean_encoder_out), self.init_c(mean_encoder_out)

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        h, c = self.init_hidden_state(encoder_out)
        decode_lengths = [c - 1 for c in caption_lengths]
        max_len = max(decode_lengths)
        embeddings = self.embedding(encoded_captions)
        
        predictions = torch.zeros(batch_size, max_len, self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max_len, encoder_out.size(1)).to(encoder_out.device)
        
        for t in range(max_len):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1)
            h, c = self.decode_step(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            predictions[:batch_size_t, t, :] = self.fc(h)
            alphas[:batch_size_t, t, :] = alpha
            
        return predictions, alphas

# --- 4. TEST IT IN VS CODE ---
if __name__ == "__main__":
    print("Initializing Show, Attend, and Tell Model...")
    
    # 1. Setup our model parameters
    vocab_size = 5000  # Imagine our dictionary has 5000 words
    encoder = Encoder()
    decoder = DecoderWithAttention(attention_dim=512, embed_dim=512, decoder_dim=512, vocab_size=vocab_size)
    
    # 2. Create some "Fake" Data to test if the code works
    # Fake image batch: 2 images, 3 color channels, 256x256 pixels
    dummy_images = torch.randn(2, 3, 256, 256) 
    # Fake captions: 2 sentences, padded to a max length of 15 words
    dummy_captions = torch.randint(0, vocab_size, (2, 15)) 
    # Lengths of those two fake sentences
    dummy_caption_lengths = [12, 10] 

    print("Passing fake images through the Encoder ('Show')...")
    encoded_images = encoder(dummy_images)
    print(f"Encoded Images Shape: {encoded_images.shape}") # Should be [2, 196, 2048]

    print("Passing encoded images and captions to the Decoder ('Attend and Tell')...")
    predictions, attention_weights = decoder(encoded_images, dummy_captions, dummy_caption_lengths)
    
    print("\nSUCCESS! The model processed the data.")
    print(f"Predictions Shape: {predictions.shape} -> (Batch Size, Max Sentence Length, Vocab Size)")
    print(f"Attention Weights Shape: {attention_weights.shape} -> (Batch Size, Max Sentence Length, Pixels)")