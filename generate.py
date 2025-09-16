import torch
import random
import argparse

from model import LSTMMusicGenerator
from utils import notes_to_midi

def load_trained_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    
    hyperparams = checkpoint['hyperparameters']
    processor = checkpoint['processor']
    
    model = LSTMMusicGenerator(
        vocab_size=hyperparams['vocab_size'],
        embedding_dim=hyperparams['embedding_dim'],
        hidden_dim=hyperparams['hidden_dim'],
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, processor

def generate_music(model, processor, seed_sequence, length, temperature, device):
    model.eval()
    
    current_sequence = seed_sequence.copy()
    generated_notes = []
    
    with torch.no_grad():
        for _ in range(length):
            seq_tensor = torch.LongTensor([current_sequence[-processor.sequence_length:]]).to(device)
            
            output, _ = model(seq_tensor)
            
            output = output.squeeze() / temperature
            probabilities = torch.softmax(output, dim=0)
            
            next_note_idx = torch.multinomial(probabilities, 1).item()
            
            generated_notes.append(processor.int_to_note[next_note_idx])
            current_sequence.append(next_note_idx)
            
    return generated_notes

def main():
    parser = argparse.ArgumentParser(description="Generate music using a trained LSTM model.")
    parser.add_argument('--model_path', type=str, default='lstm_music_generator.pth', help='Path to the trained model file.')
    parser.add_argument('--length', type=int, default=200, help='Number of notes to generate.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Controls randomness. Higher is more random.')
    parser.add_argument('--output_file', type=str, default='generated_music.mid', help='Name of the output MIDI file.')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    try:
        model, processor = load_trained_model(args.model_path, device)
    except FileNotFoundError:
        print(f"Error: Model file not found at '{args.model_path}'. Please train the model first by running train.py.")
        return
        
    print("Generating music...")

    seed_sequence = [random.randint(0, processor.vocab_size - 1) for _ in range(processor.sequence_length)]
    
    generated_notes = generate_music(
        model, 
        processor, 
        seed_sequence, 
        length=args.length, 
        temperature=args.temperature, 
        device=device
    )
    
    notes_to_midi(generated_notes, args.output_file)

if __name__ == '__main__':
    main()
