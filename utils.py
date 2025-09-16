import pretty_midi
import matplotlib.pyplot as plt

def notes_to_midi(notes, output_file, tempo=120):
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=0)  

    current_time = 0.0
    note_duration = 0.5  

    for note_pitch in notes:
        if isinstance(note_pitch, (int, float)) and 0 <= note_pitch <= 127:
            note = pretty_midi.Note(
                velocity=100,
                pitch=int(note_pitch),
                start=current_time,
                end=current_time + note_duration
            )
            instrument.notes.append(note)
            current_time += 0.3 
            
    midi.instruments.append(instrument)
    midi.write(output_file)
    print(f"MIDI file saved to {output_file}")


def plot_training_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()
