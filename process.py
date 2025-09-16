import os
import pretty_midi
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class MIDIProcessor:
    def __init__(self, sequence_length=100):
        self.sequence_length = sequence_length
        self.note_to_int = {}
        self.int_to_note = {}
        self.vocab_size = 0

    def extract_notes_from_midi(self, midi_file):
        try:
            midi = pretty_midi.PrettyMIDI(midi_file)
            notes = []
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        notes.append(note.pitch)
            return notes
        except Exception as e:
            print(f"Error processing {midi_file}: {e}")
            return []

    def process_dataset(self, midi_files, max_files=None):
        all_notes = []
        if max_files:
            midi_files = midi_files[:max_files]

        print(f"Processing {len(midi_files)} MIDI files...")
        for midi_file in tqdm(midi_files):
            notes = self.extract_notes_from_midi(midi_file)
            if notes:
                all_notes.extend(notes)

        if not all_notes:
            return []

        unique_notes = sorted(set(all_notes))
        self.note_to_int = {note: i for i, note in enumerate(unique_notes)}
        self.int_to_note = {i: note for i, note in enumerate(unique_notes)}
        self.vocab_size = len(unique_notes)

        print(f"Vocabulary size: {self.vocab_size}")
        return all_notes

    def create_sequences(self, notes):
        sequences = []
        targets = []
        for i in range(len(notes) - self.sequence_length):
            seq = notes[i:i + self.sequence_length]
            target = notes[i + self.sequence_length]
            
            seq_int = [self.note_to_int.get(note) for note in seq]
            target_int = self.note_to_int.get(target)
            
            if None not in seq_int and target_int is not None:
                sequences.append(seq_int)
                targets.append(target_int)
                
        return np.array(sequences), np.array(targets)


class MusicDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.LongTensor(sequences)
        self.targets = torch.LongTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def find_midi_files(directory):
    midi_files = []
    if os.path.exists(directory):
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(('.mid', '.midi')):
                    midi_files.append(os.path.join(root, file))
    return midi_files
