# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

# Project imports
from data_processing import MIDIProcessor, MusicDataset, find_midi_files
from model import LSTMMusicGenerator
from utils import plot_training_loss, notes_to_midi

# --- Configuration ---
SEQUENCE_LENGTH = 100
BATCH_SIZE = 64
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.3
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
MAX_FILES = 200  # Limit files for faster training/memory constraints. Set to None to use all files.
DATASET_DIR = "./maestro-v3.0.0"
MODEL_SAVE_PATH = "lstm_music_generator.pth"

def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    """Main training loop."""
    model.to(device)
    model.train()
    train_losses = []

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for sequences, targets in progress_bar:
            sequences, targets = sequences.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')

        gc.collect()
        torch.cuda.empty_cache()

    return train_losses

def main():
    """Orchestrates the data processing, model training, and saving."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Data Processing
    midi_files = find_midi_files(DATASET_DIR)
    if not midi_files:
        print(f"No MIDI files found in '{DATASET_DIR}'. Please run download_dataset.sh")
        return

    processor = MIDIProcessor(SEQUENCE_LENGTH)
    all_notes = processor.process_dataset(midi_files, max_files=MAX_FILES)
    if not all_notes:
        print("No notes were extracted from the MIDI files. Exiting.")
        return

    sequences, targets = processor.create_sequences(all_notes)
    print(f"Created {len(sequences)} training sequences.")

    dataset = MusicDataset(sequences, targets)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Model Initialization
    model = LSTMMusicGenerator(
        vocab_size=processor.vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 3. Training
    train_losses = train_model(model, dataloader, criterion, optimizer, NUM_EPOCHS, device)
    plot_training_loss(train_losses)
    
    # 4. Save the final model and processor
    torch.save({
        'model_state_dict': model.state_dict(),
        'processor': processor,
        'hyperparameters': {
            'vocab_size': processor.vocab_size,
            'embedding_dim': EMBEDDING_DIM,
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'sequence_length': SEQUENCE_LENGTH
        }
    }, MODEL_SAVE_PATH)
    print(f"\nTraining complete. Model saved to '{MODEL_SAVE_PATH}'")

if __name__ == '__main__':
    main()
