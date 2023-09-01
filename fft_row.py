import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Load CSV data into a pandas DataFrame
csv_file = 'drone.csv'
df = pd.read_csv(csv_file)

# Extract gyroscope and acceleration data from the DataFrame
gyro_x_data = df['gyrox'].values* 0.060975609756
gyro_y_data = df['gyroy'].values* 0.060975609756
gyro_z_data = df['gyroz'].values* 0.060975609756
accel_x_data = df['accx'].values*0.0004882813
accel_y_data = df['accy'].values*0.0004882813
accel_z_data = df['accz'].values*0.0004882813

# Sample frequency and time period
sample_rate =200 # Calculate sample rate based on time column
time_period = 1 / sample_rate

# Compute FFT for accelerometer and gyroscope data
accel_x_fft = fft(accel_x_data)
accel_y_fft = fft(accel_y_data)
accel_z_fft = fft(accel_z_data)
gyro_x_fft = fft(gyro_x_data)
gyro_y_fft = fft(gyro_y_data)
gyro_z_fft = fft(gyro_z_data)

# Frequency values for plotting (positive frequencies only)
n = len(accel_x_data)
positive_freqs = np.fft.fftfreq(n, time_period)[:n//2]

# Filter out frequencies under 10 Hz
min_frequency = 10  # Minimum frequency in Hz
indices_to_keep = positive_freqs >= min_frequency
accel_x_fft = accel_x_fft[:n//2][indices_to_keep]
accel_y_fft = accel_y_fft[:n//2][indices_to_keep]
accel_z_fft = accel_z_fft[:n//2][indices_to_keep]
gyro_x_fft = gyro_x_fft[:n//2][indices_to_keep]
gyro_y_fft = gyro_y_fft[:n//2][indices_to_keep]
gyro_z_fft = gyro_z_fft[:n//2][indices_to_keep]
positive_freqs = positive_freqs[indices_to_keep]

# Create subplots
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
plt.subplots_adjust(wspace=0.4)

# Plot accelerometer data
axs[0, 0].plot(positive_freqs, np.abs(accel_x_fft), label='Accel X', linewidth=0.5)
axs[0, 0].set_title('Accelerometer X FFT')
axs[0, 0].set_xlabel('Frequency (Hz)')
axs[0, 0].set_ylabel('Magnitude')
axs[0, 0].legend()

axs[0, 1].plot(positive_freqs, np.abs(accel_y_fft), label='Accel Y', linewidth=0.5)
axs[0, 1].set_title('Accelerometer Y FFT')
axs[0, 1].set_xlabel('Frequency (Hz)')
axs[0, 1].set_ylabel('Magnitude')
axs[0, 1].legend()

axs[0, 2].plot(positive_freqs, np.abs(accel_z_fft), label='Accel Z', linewidth=0.5)
axs[0, 2].set_title('Accelerometer Z FFT')
axs[0, 2].set_xlabel('Frequency (Hz)')
axs[0, 2].set_ylabel('Magnitude')
axs[0, 2].legend()

# Plot gyroscope data
axs[1, 0].plot(positive_freqs, np.abs(gyro_x_fft), label='Gyro X', linewidth=0.5)
axs[1, 0].set_title('Gyroscope X FFT')
axs[1, 0].set_xlabel('Frequency (Hz)')
axs[1, 0].set_ylabel('Magnitude')
axs[1, 0].legend()

axs[1, 1].plot(positive_freqs, np.abs(gyro_y_fft), label='Gyro Y', linewidth=0.5)
axs[1, 1].set_title('Gyroscope Y FFT')
axs[1, 1].set_xlabel('Frequency (Hz)')
axs[1, 1].set_ylabel('Magnitude')
axs[1, 1].legend()

axs[1, 2].plot(positive_freqs, np.abs(gyro_z_fft), label='Gyro Z', linewidth=0.5)
axs[1, 2].set_title('Gyroscope Z FFT')
axs[1, 2].set_xlabel('Frequency (Hz)')
axs[1, 2].set_ylabel('Magnitude')
axs[1, 2].legend()

plt.tight_layout()
plt.show()