%% Generate Sample PPG Data for Testing
% This script generates a sample PPG signal and saves it as ppg.csv

clear all; close all; clc;

fprintf('Generating sample PPG signal...\n');

% Sampling parameters
fs = 100; % Sampling frequency (Hz)
duration = 120; % Signal duration (seconds)
t = 0:1/fs:duration;

% Generate realistic PPG signal components
% Heart rate component (around 1.2 Hz = 72 bpm)
heart_rate = 1.2; % Hz
ppg_cardiac = sin(2*pi*heart_rate*t);

% Respiratory component (around 0.25 Hz = 15 breaths/min)
resp_rate = 0.25; % Hz
ppg_respiratory = 0.3 * sin(2*pi*resp_rate*t);

% Low frequency component (Mayer waves around 0.1 Hz)
mayer_freq = 0.1; % Hz
ppg_mayer = 0.2 * sin(2*pi*mayer_freq*t);

% Higher frequency harmonics
harmonic1 = 0.2 * sin(2*pi*2*heart_rate*t);
harmonic2 = 0.1 * sin(2*pi*3*heart_rate*t);

% Add some physiological variability
variability = 0.05 * sin(2*pi*0.04*t) + 0.03 * sin(2*pi*0.08*t);

% Combine all components
PPG1 = ppg_cardiac + ppg_respiratory + ppg_mayer + harmonic1 + harmonic2 + variability;

% Add realistic noise
noise_level = 0.05;
PPG1 = PPG1 + noise_level * randn(size(t));

% Add baseline drift
baseline_drift = 0.1 * sin(2*pi*0.01*t);
PPG1 = PPG1 + baseline_drift;

% Normalize signal
PPG1 = (PPG1 - mean(PPG1)) / std(PPG1);

% Scale to realistic amplitude range (arbitrary units)
PPG1 = 1000 + 100 * PPG1;

% Create additional PPG channels with slight variations (optional)
PPG2 = 1000 + 100 * (ppg_cardiac + 0.8*ppg_respiratory + 0.15*ppg_mayer + ...
       0.15*harmonic1 + 0.08*harmonic2 + 0.04*randn(size(t)));

% Create time column
Time = t';
PPG1 = PPG1';
PPG2 = PPG2';

% Create table
ppg_table = table(Time, PPG1, PPG2);

% Save to CSV file
filename = 'ppg.csv';
writetable(ppg_table, filename);

fprintf('Sample PPG data saved to %s\n', filename);
fprintf('Signal properties:\n');
fprintf('  - Duration: %d seconds\n', duration);
fprintf('  - Sampling rate: %d Hz\n', fs);
fprintf('  - Number of samples: %d\n', length(PPG1));
fprintf('  - Main frequency components:\n');
fprintf('    * Heart rate: %.2f Hz (%.0f bpm)\n', heart_rate, heart_rate*60);
fprintf('    * Respiratory: %.2f Hz\n', resp_rate);
fprintf('    * Mayer waves: %.2f Hz\n', mayer_freq);

% Plot the generated signal
figure('Position', [100, 100, 1200, 600]);

% Plot first 30 seconds
plot_duration = min(30, duration);
plot_samples = plot_duration * fs;

subplot(2, 1, 1);
plot(Time(1:plot_samples), PPG1(1:plot_samples), 'b-', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('PPG Amplitude');
title('Generated PPG Signal (First 30 seconds)');
grid on;

% Plot frequency spectrum
subplot(2, 1, 2);
[pxx, f] = pwelch(PPG1, hamming(1024), 512, 2048, fs);
plot(f, 10*log10(pxx), 'r-', 'LineWidth', 1);
xlim([0, 5]);
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density (dB)');
title('Power Spectrum of Generated PPG Signal');
grid on;

fprintf('Sample PPG data generation complete!\n');