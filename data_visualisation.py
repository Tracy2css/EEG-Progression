# %%
"""Example EEG preprocessing of EMOTIV data
This example script demonstrates a few basic functions to import and annotate EEG data collected from EmotivPRO software. It uses MNE to load an XDF file, print some basic metadata, create an `info` object and plot the power spectrum."""

import pyxdf
import mne
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from mne.time_frequency import tfr_multitaper
import warnings

# %%
# Path to your XDF file
# data_path = r"F:\360MoveData\CurrentStudy\sub-P003\ses-S001\eeg\sub-P003_ses-S001_task-Default_run-001_eeg.xdf"
data_path = r"F:\360MoveData\CurrentStudy\sub-P002\ses-S001\eeg\sub-P002_ses-S001_task-Default_run-001_eeg.xdf"

# Load the XDF file
streams, fileheader = pyxdf.load_xdf(data_path)
print("XDF File Header:", fileheader)
print("Number of streams found:", len(streams))

# Ensure there are at least two streamsPP
if len(streams) < 2:
    print("Error: Less than two streams found.")
else:
    # Directly extract the second stream
    stream = streams[1]

    # Extract data from the second stream
    data = np.array(stream['time_series']).T
    timestamps = np.array(stream['time_stamps'])
    channel_names = [chan['label'][0] for chan in stream['info']['desc'][0]['channels'][0]['channel']]
    sfreq = float(stream['info']['nominal_srate'][0])
    channel_types = ['eeg'] * len(channel_names)  # Assuming all channels are EEG channels

    # Print the extracted data
    print("\nExtracted Stream 2 Data:")
    print("Stream Name:", stream['info']['name'][0])
    print("Stream Type:", stream['info']['type'][0])
    print("Number of Channels:", stream['info']['channel_count'][0])
    print("Sampling Rate:", float(stream['info']['nominal_srate'][0]))
    print("Number of Samples:", len(stream['time_series']))
    print("First 5 data points:", stream['time_series'][:5])
    print("Channel Names:", channel_names)


# %%
# Extract and transpose data (channels x samples)
data = np.array(stream['time_series']).T

# Handle zero and infinite values
data[np.isinf(data)] = np.nan  # Replace infinite values with NaN
data = np.nan_to_num(data, nan=0.0)  # Replace NaN with zero

# Remove problematic and non-EEG channels
bad_channels = ['Interpolate', 'HardwareMarker', 'Markers', 'Timestamp', 'Counter']
picks = [i for i, name in enumerate(channel_names) if name not in bad_channels]
data = data[picks, :]
channel_names = [channel_names[i] for i in picks]

# Create MNE info object
info = mne.create_info(channel_names, sfreq, channel_types)

# Create RawArray object for raw data
raw = mne.io.RawArray(data, info)

# %%
# Create a copy of the raw data for filtering
raw_filtered = raw.copy()

# Set standard montage for channel locations
montage = mne.channels.make_standard_montage('standard_1020')
raw_filtered.set_montage(montage)  # Ensure montage is set

# Apply high-pass filter to remove DC offset and baseline drift
raw_filtered.filter(1., None, fir_design='firwin')

# Apply band-pass filter to remove low and high frequency noise
raw_filtered.filter(1., 50., fir_design='firwin')

# Apply notch filter to remove power line noise
raw_filtered.notch_filter(np.arange(50, sfreq / 2, 50), fir_design='firwin')

# Remove linear trend to further reduce baseline drift
raw_filtered.apply_function(lambda x: mne.filter.detrend(x, axis=-1))

# Re-reference the data to average
raw_filtered.set_eeg_reference('average', projection=True)
raw_filtered.apply_proj()

# %%
# Compute power spectral density (PSD) for original and filtered data
psd_original = raw.compute_psd(fmax=50)
psd_filtered = raw_filtered.compute_psd(fmax=50)

# Get PSD data and frequencies
psd_data_original, freqs = psd_original.get_data(return_freqs=True)
psd_data_filtered, freqs = psd_filtered.get_data(return_freqs=True)

# %%
# Plot original and filtered PSD on the same plot
plt.figure()
plt.plot(freqs, np.mean(psd_data_original, axis=0), label='Original EEG Signal', color='blue')
plt.plot(freqs, np.mean(psd_data_filtered, axis=0), label='Filtered EEG Signal', color='green')
plt.title('Original and Filtered EEG Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (µV²/Hz)')
plt.legend()
plt.show()

# %%

# Plot only filtered PSD
plt.figure()
plt.plot(freqs, np.mean(psd_data_filtered, axis=0), label='Filtered EEG Signal', color='green')
plt.title('Filtered EEG Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (µV²/Hz)')
plt.legend()
plt.show()

# %%
# Plot only original PSD
plt.figure()
plt.plot(freqs, np.mean(psd_data_original, axis=0), label='Original EEG Signal', color='blue')
plt.title('Original EEG Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (µV²/Hz)')
plt.legend()
plt.show()

# %%
# Define frequency range for time-frequency analysis
freqs = np.arange(1, 50, 1)  # 1 to 50 Hz
n_cycles = freqs / 2.  # Number of cycles in each frequency

# Compute time-frequency representation using multitaper method
power = tfr_multitaper(raw_filtered, freqs=freqs, n_cycles=n_cycles, time_bandwidth=2.0, return_itc=False)

# Plot the time-frequency representation
power.plot([0], baseline=(None, 0), mode='logratio', title='Time-Frequency Representation')

# %% [markdown]
# ## 1.Low-Frequency Band (0-10 Hz):
# - The low-frequency band shows relatively higher power, especially in the 0-10 Hz range. 
# - This may correspond to delta and theta wave activity, which are typically associated with sleep, deep relaxation, or certain pathological states.
# 
# ## 2.Mid-Frequency Band (10-20 Hz):
# - The mid-frequency band exhibits more complex power changes, especially in the 10-20 Hz range. 
# - This may correspond to alpha and low beta wave activity. 
# - Alpha waves are typically prominent during relaxed, awake states, while beta waves are associated with alertness and cognitive activity.
# 
# ## 3.High-Frequency Band (20-50 Hz):
# - The high-frequency band shows relatively fewer power changes and remains mostly stable. 
# - These frequencies correspond to high beta and gamma waves, often associated with high cognitive load and information processing.


