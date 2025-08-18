%% PPG Signal Analysis
% This script analyzes PPG signal from ppg.csv file
% Focuses on 0.2-4Hz frequency band with 0.3Hz interval energy comparison

clear all; close all; clc;

%% 1. Load PPG data from CSV file
fprintf('Loading PPG data from ppg.csv...\n');
try
    % Read CSV file
    data = readtable('ppg.csv');
    
    % Extract PPG1 column
    if ismember('PPG1', data.Properties.VariableNames)
        ppg_signal = data.PPG1;
    else
        error('PPG1 column not found in the CSV file');
    end
    
    % Sampling frequency (assuming 100 Hz, adjust if different)
    fs = 100; % Hz - please adjust this according to your actual sampling rate
    
catch ME
    fprintf('Error loading data: %s\n', ME.message);
    fprintf('Creating sample PPG data for demonstration...\n');
    
    % Create sample PPG signal for demonstration if file not found
    fs = 100; % Hz
    t = 0:1/fs:60; % 60 seconds of data
    ppg_signal = sin(2*pi*1.2*t) + 0.5*sin(2*pi*2.5*t) + 0.3*sin(2*pi*0.3*t) + 0.1*randn(size(t));
    ppg_signal = ppg_signal';
end

% Time vector
t = (0:length(ppg_signal)-1) / fs;

%% 2. Skip first 5 seconds of signal
skip_time = 5; % seconds
skip_samples = round(skip_time * fs);

if length(ppg_signal) > skip_samples
    ppg_signal = ppg_signal(skip_samples+1:end);
    t = t(skip_samples+1:end);
    fprintf('Skipped first %d seconds of signal\n', skip_time);
else
    warning('Signal is shorter than 5 seconds, using entire signal');
end

%% 3. Apply bandpass filter (0.2-4 Hz)
fprintf('Applying bandpass filter (0.2-4 Hz)...\n');

% Design bandpass filter
nyquist = fs/2;
low_freq = 0.2; % Hz
high_freq = 4.0; % Hz

% Use Butterworth filter
filter_order = 4;
[b, a] = butter(filter_order, [low_freq high_freq]/nyquist, 'bandpass');

% Apply filter
ppg_filtered = filtfilt(b, a, ppg_signal);

%% 4. Time-Frequency Analysis using Short-Time Fourier Transform (STFT)
fprintf('Performing time-frequency analysis...\n');

% STFT parameters
window_length = round(10 * fs); % 10 second window
overlap = round(0.9 * window_length); % 90% overlap
nfft = 2^nextpow2(window_length * 4); % Zero padding for better frequency resolution

% Perform STFT
[S, F, T, P] = spectrogram(ppg_filtered, hamming(window_length), overlap, nfft, fs);

% Find indices for frequency range of interest (0.2-4 Hz)
freq_indices = find(F >= 0.2 & F <= 4);
F_roi = F(freq_indices);
P_roi = P(freq_indices, :);

%% 5. Define frequency bands with 0.3 Hz intervals
fprintf('Calculating energy in frequency bands...\n');

% Define frequency bands
freq_bands = [];
band_labels = {};
start_freq = 0.2;
band_width = 0.3;
end_freq = 4.0;

% Create overlapping bands
current_start = start_freq;
band_idx = 1;
while current_start <= (end_freq - band_width)
    freq_bands(band_idx, :) = [current_start, current_start + band_width];
    band_labels{band_idx} = sprintf('%.1f-%.1f Hz', current_start, current_start + band_width);
    current_start = current_start + 0.3; % Move by 0.3 Hz for next band
    band_idx = band_idx + 1;
end

% Add final band if needed to reach 4 Hz
if freq_bands(end, 2) < end_freq
    freq_bands(band_idx, :) = [freq_bands(end, 1) + 0.3, end_freq];
    band_labels{band_idx} = sprintf('%.1f-%.1f Hz', freq_bands(end, 1), end_freq);
end

num_bands = size(freq_bands, 1);

%% 6. Calculate energy for each frequency band over time
band_energy = zeros(num_bands, length(T));

for i = 1:num_bands
    % Find frequency indices for current band
    band_freq_idx = find(F_roi >= freq_bands(i, 1) & F_roi <= freq_bands(i, 2));
    
    if ~isempty(band_freq_idx)
        % Calculate energy as sum of power in the band
        band_energy(i, :) = sum(P_roi(band_freq_idx, :), 1);
    end
end

% Normalize energy for better visualization
band_energy_normalized = band_energy ./ max(band_energy, [], 2);

%% 7. Visualization

% Figure 1: Original and filtered signal
figure('Position', [100, 100, 1200, 800], 'Name', 'PPG Signal Analysis');

subplot(3, 1, 1);
plot(t, ppg_signal, 'b-', 'LineWidth', 0.5);
hold on;
plot(t, ppg_filtered, 'r-', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('PPG Signal (Blue: Original, Red: Filtered 0.2-4 Hz)');
legend('Original', 'Filtered', 'Location', 'best');
grid on;

% Figure 2: Spectrogram
subplot(3, 1, 2);
imagesc(T, F_roi, 10*log10(P_roi));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Time-Frequency Representation (Spectrogram)');
colorbar;
colormap('jet');
ylim([0.2, 4]);
caxis([-40, max(10*log10(P_roi(:)))]);

% Figure 3: Band energy over time
subplot(3, 1, 3);
% Create a color map for different bands
colors = lines(num_bands);
hold on;
for i = 1:num_bands
    plot(T, band_energy_normalized(i, :), 'LineWidth', 1.5, 'Color', colors(i, :), ...
         'DisplayName', band_labels{i});
end
xlabel('Time (s)');
ylabel('Normalized Energy');
title('Energy in Different Frequency Bands (0.3 Hz intervals)');
legend('Location', 'eastoutside', 'FontSize', 8);
grid on;
hold off;

%% 8. Additional Analysis - Energy Comparison
figure('Position', [100, 100, 1200, 600], 'Name', 'Frequency Band Energy Comparison');

% Calculate mean energy for each band
mean_energy = mean(band_energy, 2);
std_energy = std(band_energy, 0, 2);

% Bar plot of mean energy
subplot(1, 2, 1);
bar(1:num_bands, mean_energy, 'FaceColor', [0.2, 0.6, 0.8]);
set(gca, 'XTick', 1:num_bands, 'XTickLabel', band_labels, 'XTickLabelRotation', 45);
xlabel('Frequency Band');
ylabel('Mean Energy');
title('Mean Energy per Frequency Band');
grid on;

% Error bar plot
subplot(1, 2, 2);
errorbar(1:num_bands, mean_energy, std_energy, 'o-', 'LineWidth', 2, ...
         'MarkerSize', 8, 'MarkerFaceColor', [0.8, 0.2, 0.2]);
set(gca, 'XTick', 1:num_bands, 'XTickLabel', band_labels, 'XTickLabelRotation', 45);
xlabel('Frequency Band');
ylabel('Energy');
title('Mean Energy with Standard Deviation');
grid on;

%% 9. Heatmap of band energy over time
figure('Position', [100, 100, 1200, 600], 'Name', 'Energy Heatmap');
imagesc(T, 1:num_bands, band_energy_normalized);
colorbar;
colormap('hot');
set(gca, 'YTick', 1:num_bands, 'YTickLabel', band_labels);
xlabel('Time (s)');
ylabel('Frequency Band');
title('Energy Distribution Across Frequency Bands Over Time');

%% 10. Statistical Summary
fprintf('\n========== Statistical Summary ==========\n');
fprintf('Signal duration after skipping: %.2f seconds\n', length(ppg_filtered)/fs);
fprintf('Number of frequency bands: %d\n', num_bands);
fprintf('\nEnergy Statistics per Band:\n');
fprintf('%-15s | %-12s | %-12s | %-12s\n', 'Band', 'Mean Energy', 'Std Dev', 'Max Energy');
fprintf('------------------------------------------------\n');

for i = 1:num_bands
    fprintf('%-15s | %12.2e | %12.2e | %12.2e\n', ...
            band_labels{i}, mean_energy(i), std_energy(i), max(band_energy(i, :)));
end

% Find dominant frequency band
[~, dominant_band_idx] = max(mean_energy);
fprintf('\nDominant frequency band: %s\n', band_labels{dominant_band_idx});

%% 11. Save results
fprintf('\nSaving analysis results...\n');

% Save processed data
results.filtered_signal = ppg_filtered;
results.time = t;
results.frequency_bands = freq_bands;
results.band_labels = band_labels;
results.band_energy = band_energy;
results.band_energy_normalized = band_energy_normalized;
results.mean_energy = mean_energy;
results.std_energy = std_energy;
results.spectrogram_time = T;
results.spectrogram_freq = F_roi;
results.spectrogram_power = P_roi;

save('ppg_analysis_results.mat', 'results');
fprintf('Results saved to ppg_analysis_results.mat\n');

% Export energy data to CSV
energy_table = array2table([T', band_energy'], ...
    'VariableNames', ['Time', band_labels]);
writetable(energy_table, 'band_energy_results.csv');
fprintf('Band energy data saved to band_energy_results.csv\n');

fprintf('\nAnalysis complete!\n');