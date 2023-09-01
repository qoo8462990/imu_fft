import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Load CSV data into a pandas DataFrame
csv_file = 'structure.csv'
df = pd.read_csv(csv_file)

# Extract gyroscope and acceleration data from the DataFrame
gyro_x_data = df['gyrox'].values
gyro_y_data = df['gyroy'].values
gyro_z_data = df['gyroz'].values
accel_x_data = df['accx'].values    
accel_y_data = df['accy'].values
accel_z_data = df['accz'].values

# Sample frequency and time period
time_period = (df['sys_time'][1] - df['sys_time'][0]) *0.001
sample_rate = 1/ time_period

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

# Plot accelerometer data
plt.figure(figsize=(10, 12))

plt.subplot(2, 1, 1)
plt.plot(positive_freqs, np.abs(accel_x_fft)/ len(accel_x_data) / 2048, label='Accel X', linewidth=0.3)
plt.plot(positive_freqs, np.abs(accel_y_fft)/ len(accel_y_data) / 2048, label='Accel Y', linewidth=0.3)
plt.plot(positive_freqs, np.abs(accel_z_fft)/ len(accel_z_data) / 2048, label='Accel Z', linewidth=0.3)
plt.title('Accelerometer FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()

# Plot gyroscope data
plt.subplot(2, 1, 2)
plt.plot(positive_freqs, np.abs(gyro_x_fft)/ len(gyro_x_data) * 0.060975609756, label='Gyro X', linewidth=0.3)
plt.plot(positive_freqs, np.abs(gyro_y_fft)/ len(gyro_y_data) * 0.060975609756, label='Gyro Y', linewidth=0.3)
plt.plot(positive_freqs, np.abs(gyro_z_fft)/ len(gyro_z_data) * 0.060975609756, label='Gyro Z', linewidth=0.3)
plt.title('Gyroscope FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.legend()

plt.tight_layout(h_pad=2.5)  # Adjust the vertical spacing between subplots
plt.show()