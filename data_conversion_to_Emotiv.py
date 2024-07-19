import pyxdf
import numpy as np
import pyedflib

# Load the XDF file
data, header = pyxdf.load_xdf(r"F:\360MoveData\CurrentStudy\sub-P003\ses-S001\eeg\sub-P003_ses-S001_task-Default_run-001_eeg.xdf")

# Assuming the EEG data stream is the first stream
eeg_stream = data[0]
samples = np.array(eeg_stream['time_series'])
timestamps = np.array(eeg_stream['time_stamps'])

# Create an EDF file
edf_file = pyedflib.EdfWriter('sub-P003_ses-S001_task-Default_run-001_eeg.edf', len(samples[0]), file_type=pyedflib.FILETYPE_EDFPLUS)

# Set channel information
channel_info = [{'label': f'Channel {i+1}', 'dimension': 'uV', 'sample_rate': 128, 'physical_min': np.min(samples), 'physical_max': np.max(samples), 'digital_min': -32768, 'digital_max': 32767, 'transducer': '', 'prefilter': ''} for i in range(samples.shape[1])]

edf_file.setSignalHeaders(channel_info)

# Write data to EDF file
edf_file.writeSamples([samples[:, i] for i in range(samples.shape[1])])
edf_file.close()
