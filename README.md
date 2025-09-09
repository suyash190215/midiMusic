# LSTM-Based Generative Music Model

This project features an **LSTM-based music generation model** built using PyTorch. The model was trained on the **Maestro v3.0.0 dataset**, which consists of over 1,200 classical piano pieces in MIDI format. Our goal was to create a system that could automatically compose original musical pieces with realistic note transitions.

***

## Key Features 

* **Advanced Data Processing:** We implemented efficient data pipelines that utilize **sequence padding** and **batching** with a batch size of 64. This optimization led to a **2x increase in training speed** compared to standard baselines.
* **Optimized LSTM Architecture:** The model uses a two-layer LSTM with **256 hidden units** and a **dropout rate of 0.3**. Through careful hyperparameter tuning, we were able to **reduce the cross-entropy loss by 20%**, resulting in higher-quality generated music.
* **Automated Composition:** The model was trained for 10 epochs on a **P100 GPU** via Google Colab. It can generate original music compositions that are over two minutes long, showcasing smooth and realistic transitions between notes.

***

## How It Works

The model processes MIDI files from the Maestro dataset, converting them into a numerical format suitable for deep learning. The **LSTM (Long Short-Term Memory)** network is then trained to predict the next note in a sequence based on the preceding notes. This allows it to learn the complex temporal relationships and patterns present in classical music.



After training, the model can be "primed" with a starting note or a short sequence. It then uses its learned patterns to generate a new, original musical composition note by note. The generated output can be saved as a MIDI file, which can then be played using any standard music software.

***

## Technologies Used 

* **PyTorch**: The primary deep learning framework.
* **Google Colab**: Used for its access to powerful GPUs (specifically the P100) to accelerate training.
* **Maestro v3.0.0 Dataset**: The comprehensive dataset of classical piano pieces used for training.
