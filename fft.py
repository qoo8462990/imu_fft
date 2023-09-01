import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft
from matplotlib.pyplot import specgram

# Load CSV data into a pandas DataFrame
csv_file = 'drone.csv'
df = pd.read_csv(csv_file)

# Extract data from the DataFrame
time_data = df['sys_time'].values
accel_x_data = df['accx'].values * 0.0004882813
accel_y_data = df['accy'].values * 0.0004882813
accel_z_data = df['accz'].values * 0.0004882813
gyro_x_data = df['gyrox'].values * 0.060975609756
gyro_y_data = df['gyroy'].values * 0.060975609756
gyro_z_data = df['gyroz'].values * 0.060975609756

# Sample frequency and time period
time_period = (time_data[1] - time_data[0]) * 0.001
sample_rate = 1 / time_period

# Create a list of data and labels for iteration
accel_data_list = [
    (accel_x_data, 'Accel X', 'Acceleration (g)'),
    (accel_y_data, 'Accel Y', 'Acceleration (g)'),
    (accel_z_data, 'Accel Z', 'Acceleration (g)')
]

gyro_data_list = [
    (gyro_x_data, 'Gyro X', 'Angular Velocity (deg/s)'),
    (gyro_y_data, 'Gyro Y', 'Angular Velocity (deg/s)'),
    (gyro_z_data, 'Gyro Z', 'Angular Velocity (deg/s)')
]

# Create subplots for accelerometer data in Figure 1
fig1, axs1 = plt.subplots(len(accel_data_list), 3, figsize=(15, 10))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
fig1.suptitle('Accelerometer Data', fontsize=16)

for i, (data, label, ylabel) in enumerate(accel_data_list):
    # Compute FFT for the current data
    fft_data = fft(data)

    # Frequency values for plotting (positive frequencies only)
    n = len(data)
    positive_freqs = np.fft.fftfreq(n, time_period)[:n // 2]

    # Filter out frequencies under 10 Hz
    min_frequency = 10  # Minimum frequency in Hz
    indices_to_keep = positive_freqs >= min_frequency
    fft_data = fft_data[:n // 2][indices_to_keep]
    positive_freqs = positive_freqs[indices_to_keep]

    # Plot raw data
    axs1[i, 0].plot(time_data, data, label=label, linewidth=0.3)
    axs1[i, 0].set_title(f'{label} Raw Data')
    axs1[i, 0].set_xlabel('Time (s)')
    axs1[i, 0].set_ylabel(ylabel)
    axs1[i, 0].legend()

    # Plot FFT data
    axs1[i, 1].plot(positive_freqs, np.abs(fft_data) / len(data), label=label, linewidth=0.3)
    axs1[i, 1].set_title(f'{label} FFT')
    axs1[i, 1].set_xlabel('Frequency (Hz)')
    axs1[i, 1].set_ylabel('Magnitude')
    axs1[i, 1].legend()

    # Plot spectrogram
    axs1[i, 2].specgram(data, Fs=sample_rate, NFFT=256, noverlap=128, cmap='viridis')
    axs1[i, 2].set_title(f'Spectrogram for {label}')
    axs1[i, 2].set_xlabel('Time (s)')
    axs1[i, 2].set_ylabel('Frequency (Hz)')

# Create subplots for gyroscope data in Figure 2
fig2, axs2 = plt.subplots(len(gyro_data_list), 3, figsize=(15, 10))
plt.subplots_adjust(wspace=0.4, hspace=0.4)
fig2.suptitle('Gyroscope Data', fontsize=16)

for i, (data, label, ylabel) in enumerate(gyro_data_list):
    # Compute FFT for the current data
    fft_data = fft(data)

    # Frequency values for plotting (positive frequencies only)
    n = len(data)
    positive_freqs = np.fft.fftfreq(n, time_period)[:n // 2]

    # Filter out frequencies under 10 Hz
    min_frequency = 10  # Minimum frequency in Hz
    indices_to_keep = positive_freqs >= min_frequency
    fft_data = fft_data[:n // 2][indices_to_keep]
    positive_freqs = positive_freqs[indices_to_keep]

    # Plot raw data
    axs2[i, 0].plot(time_data, data, label=label, linewidth=0.3)
    axs2[i, 0].set_title(f'{label} Raw Data')
    axs2[i, 0].set_xlabel('Time (s)')
    axs2[i, 0].set_ylabel(ylabel)
    axs2[i, 0].legend()

    # Plot FFT data
    axs2[i, 1].plot(positive_freqs, np.abs(fft_data) / len(data), label=label, linewidth=0.3)
    axs2[i, 1].set_title(f'{label} FFT')
    axs2[i, 1].set_xlabel('Frequency (Hz)')
    axs2[i, 1].set_ylabel('Magnitude')
    axs2[i, 1].legend()

    # Plot spectrogram
    axs2[i, 2].specgram(data, Fs=sample_rate, NFFT=256, noverlap=128, cmap='viridis')
    axs2[i, 2].set_title(f'Spectrogram for {label}')
    axs2[i, 2].set_xlabel('Time (s)')
    axs2[i, 2].set_ylabel('Frequency (Hz)')

# Save the figures
fig1.savefig('Accelerometer_Data_Plots.png')
fig2.savefig('Gyroscope_Data_Plots.png')

# Show the figures
plt.show()
