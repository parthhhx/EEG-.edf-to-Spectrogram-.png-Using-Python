import pyedflib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Read EEG data from the EDF file
f = pyedflib.EdfReader("file_path") 
num_channels = f.signals_in_file
print(num_channels)

sampling_rate = f.getSampleFrequencies()

# Define the duration of each chunk in seconds
chunk_duration = 30 #you can adjust this variable according to the need

# Calculate the number of samples in each chunk
samples_per_chunk = int(chunk_duration * sampling_rate[0])  # Assuming all channels have the same sampling rate

# Read EEG signals from all channels
eeg_signals = [f.readSignal(i) for i in range(num_channels)]

# Split the EEG signals into 2-second chunks for all channels
eeg_chunks = [np.array([eeg[i:i + samples_per_chunk] for eeg in eeg_signals]) for i in range(0, len(eeg_signals[0]), samples_per_chunk)]

# Convert each chunk into a spectrogram for each channel
spectrograms = []
for chunk in eeg_chunks:
    channel_spectrograms = []
    for channel_signal in chunk:
        _, _, Sxx = spectrogram(channel_signal, fs=sampling_rate[0], noverlap=1)
        channel_spectrograms.append(Sxx)
    spectrograms.append(np.concatenate(channel_spectrograms, axis=0))

# Plot and save the spectrograms for each 2-second chunk
for i, spectrogram_data in enumerate(spectrograms):
    plt.figure(figsize=(8, 6))
    plt.imshow(np.log1p(spectrogram_data), aspect='auto', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram - Chunk {i + 1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.savefig(f"chunk_{i + 1}_spectrogram.png")
    plt.close()

# Close the EDF file
f._close()
