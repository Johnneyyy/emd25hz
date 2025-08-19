%% PPG Signal Analysis with EMD
% This script analyzes PPG signal from ppg.csv file
% Uses EMD for band energy calculation and STFT for visualization
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

%% 5. EMD Analysis for Band Energy Calculation
fprintf('Performing EMD analysis...\n');

% Perform EMD decomposition
try
    % Use built-in EMD if available (requires Signal Processing Toolbox)
    [imfs, residual] = emd(ppg_filtered);
    fprintf('EMD decomposition completed: %d IMFs extracted\n', size(imfs, 2));
catch
    % Simple EMD implementation if built-in not available
    fprintf('Using custom EMD implementation...\n');
    [imfs, residual] = custom_emd(ppg_filtered, 10); % Max 10 IMFs
end

% Calculate instantaneous frequency for each IMF using Hilbert transform
num_imfs = size(imfs, 2);
imf_inst_freq = zeros(size(imfs));
imf_inst_amp = zeros(size(imfs));

for i = 1:num_imfs
    % Apply Hilbert transform
    analytic_signal = hilbert(imfs(:, i));
    
    % Calculate instantaneous amplitude
    imf_inst_amp(:, i) = abs(analytic_signal);
    
    % Calculate instantaneous phase
    inst_phase = unwrap(angle(analytic_signal));
    
    % Calculate instantaneous frequency (derivative of phase)
    inst_freq = diff(inst_phase) * fs / (2*pi);
    inst_freq = [inst_freq(1); inst_freq]; % Pad to maintain size
    
    % Smooth instantaneous frequency
    inst_freq = movmean(inst_freq, round(fs/10)); % 0.1 second smoothing window
    imf_inst_freq(:, i) = inst_freq;
end

%% 6. Define frequency bands with 0.3 Hz intervals
fprintf('Calculating energy in frequency bands using EMD...\n');

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

%% 7. Calculate EMD-based energy for each frequency band over time
% Create time windows for energy calculation (similar to STFT windows)
window_length = round(10 * fs); % 10 second window
overlap = round(0.9 * window_length); % 90% overlap
step = window_length - overlap;
num_windows = floor((length(ppg_filtered) - window_length) / step) + 1;

% Time points for energy calculation
T_emd = zeros(1, num_windows);
band_energy_emd = zeros(num_bands, num_windows);

for win_idx = 1:num_windows
    % Define window boundaries
    start_idx = (win_idx - 1) * step + 1;
    end_idx = start_idx + window_length - 1;
    
    if end_idx > length(ppg_filtered)
        end_idx = length(ppg_filtered);
    end
    
    % Time point for this window (center)
    T_emd(win_idx) = mean(t(start_idx:end_idx));
    
    % Calculate energy for each frequency band in this window
    for band_idx = 1:num_bands
        band_energy_sum = 0;
        
        % Check each IMF's contribution to this frequency band
        for imf_idx = 1:num_imfs
            % Get instantaneous frequency and amplitude for this window
            inst_freq_window = imf_inst_freq(start_idx:end_idx, imf_idx);
            inst_amp_window = imf_inst_amp(start_idx:end_idx, imf_idx);
            
            % Find samples within the frequency band
            in_band = (inst_freq_window >= freq_bands(band_idx, 1)) & ...
                     (inst_freq_window <= freq_bands(band_idx, 2));
            
            % Sum energy from samples in this band
            band_energy_sum = band_energy_sum + sum(inst_amp_window(in_band).^2);
        end
        
        band_energy_emd(band_idx, win_idx) = band_energy_sum / window_length;
    end
end

% Normalize EMD energy for better visualization
band_energy_emd_normalized = band_energy_emd ./ max(band_energy_emd, [], 2);

%% 8. Calculate STFT-based energy for comparison
band_energy_stft = zeros(num_bands, length(T));

for i = 1:num_bands
    % Find frequency indices for current band
    band_freq_idx = find(F_roi >= freq_bands(i, 1) & F_roi <= freq_bands(i, 2));
    
    if ~isempty(band_freq_idx)
        % Calculate energy as sum of power in the band
        band_energy_stft(i, :) = sum(P_roi(band_freq_idx, :), 1);
    end
end

% Normalize STFT energy for better visualization
band_energy_stft_normalized = band_energy_stft ./ max(band_energy_stft, [], 2);

%% 9. Visualization

% Figure 1: Original and filtered signal with IMFs
figure('Position', [100, 100, 1400, 900], 'Name', 'PPG Signal Analysis with EMD');

subplot(4, 1, 1);
plot(t, ppg_signal, 'b-', 'LineWidth', 0.5);
hold on;
plot(t, ppg_filtered, 'r-', 'LineWidth', 1);
xlabel('Time (s)');
ylabel('Amplitude');
title('PPG Signal (Blue: Original, Red: Filtered 0.2-4 Hz)');
legend('Original', 'Filtered', 'Location', 'best');
grid on;

% Plot first few IMFs
subplot(4, 1, 2);
num_imfs_to_plot = min(3, num_imfs);
hold on;
for i = 1:num_imfs_to_plot
    plot(t, imfs(:, i) + (i-1)*2*std(imfs(:, i)), 'LineWidth', 1);
end
xlabel('Time (s)');
ylabel('IMFs');
title(sprintf('First %d Intrinsic Mode Functions (IMFs)', num_imfs_to_plot));
legend(arrayfun(@(x) sprintf('IMF %d', x), 1:num_imfs_to_plot, 'UniformOutput', false), ...
       'Location', 'best');
grid on;
hold off;

% Spectrogram
subplot(4, 1, 3);
imagesc(T, F_roi, 10*log10(P_roi));
axis xy;
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('Time-Frequency Representation (STFT Spectrogram)');
colorbar;
colormap('jet');
ylim([0.2, 4]);
caxis([-40, max(10*log10(P_roi(:)))]);

% EMD-based band energy over time
subplot(4, 1, 4);
colors = lines(num_bands);
hold on;
for i = 1:num_bands
    plot(T_emd, band_energy_emd_normalized(i, :), 'LineWidth', 1.5, 'Color', colors(i, :), ...
         'DisplayName', band_labels{i});
end
xlabel('Time (s)');
ylabel('Normalized Energy');
title('EMD-based Energy in Different Frequency Bands (0.3 Hz intervals)');
legend('Location', 'eastoutside', 'FontSize', 7);
grid on;
hold off;

%% 10. Additional Analysis - EMD vs STFT Energy Comparison
figure('Position', [100, 100, 1400, 800], 'Name', 'EMD vs STFT Energy Comparison');

% Calculate mean energy for each band (EMD)
mean_energy_emd = mean(band_energy_emd, 2);
std_energy_emd = std(band_energy_emd, 0, 2);

% Calculate mean energy for each band (STFT)
mean_energy_stft = mean(band_energy_stft, 2);
std_energy_stft = std(band_energy_stft, 0, 2);

% Comparison bar plot
subplot(2, 2, 1);
x = 1:num_bands;
width = 0.35;
bar(x - width/2, mean_energy_emd, width, 'FaceColor', [0.2, 0.6, 0.8], 'DisplayName', 'EMD');
hold on;
bar(x + width/2, mean_energy_stft, width, 'FaceColor', [0.8, 0.2, 0.2], 'DisplayName', 'STFT');
set(gca, 'XTick', 1:num_bands, 'XTickLabel', band_labels, 'XTickLabelRotation', 45);
xlabel('Frequency Band');
ylabel('Mean Energy');
title('Mean Energy Comparison: EMD vs STFT');
legend('Location', 'best');
grid on;
hold off;

% Error bar plot for EMD
subplot(2, 2, 2);
errorbar(1:num_bands, mean_energy_emd, std_energy_emd, 'o-', 'LineWidth', 2, ...
         'MarkerSize', 8, 'MarkerFaceColor', [0.2, 0.6, 0.8]);
set(gca, 'XTick', 1:num_bands, 'XTickLabel', band_labels, 'XTickLabelRotation', 45);
xlabel('Frequency Band');
ylabel('Energy');
title('EMD: Mean Energy with Standard Deviation');
grid on;

% Correlation between EMD and STFT energies
subplot(2, 2, 3);
scatter(mean_energy_stft, mean_energy_emd, 100, 'filled');
hold on;
% Add diagonal line
max_val = max([mean_energy_stft; mean_energy_emd]);
plot([0, max_val], [0, max_val], 'r--', 'LineWidth', 1);
xlabel('STFT Mean Energy');
ylabel('EMD Mean Energy');
title('Correlation: EMD vs STFT Energy');
grid on;
% Calculate and display correlation coefficient
corr_coef = corrcoef(mean_energy_stft, mean_energy_emd);
text(0.1*max_val, 0.9*max_val, sprintf('R = %.3f', corr_coef(1,2)), 'FontSize', 12);
hold off;

% Time series comparison for selected bands
subplot(2, 2, 4);
bands_to_plot = [1, round(num_bands/2), num_bands]; % First, middle, last bands
hold on;
for i = 1:length(bands_to_plot)
    band_idx = bands_to_plot(i);
    plot(T_emd, band_energy_emd_normalized(band_idx, :), '-', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('EMD: %s', band_labels{band_idx}));
    % Interpolate STFT to match EMD time points for comparison
    stft_interp = interp1(T, band_energy_stft_normalized(band_idx, :), T_emd, 'linear', 'extrap');
    plot(T_emd, stft_interp, '--', 'LineWidth', 1.5, ...
         'DisplayName', sprintf('STFT: %s', band_labels{band_idx}));
end
xlabel('Time (s)');
ylabel('Normalized Energy');
title('Time Series Comparison: EMD (solid) vs STFT (dashed)');
legend('Location', 'best', 'FontSize', 8);
grid on;
hold off;

%% 11. Heatmap of band energy over time
figure('Position', [100, 100, 1400, 600], 'Name', 'Energy Heatmap Comparison');

% EMD heatmap
subplot(1, 2, 1);
imagesc(T_emd, 1:num_bands, band_energy_emd_normalized);
colorbar;
colormap('hot');
set(gca, 'YTick', 1:num_bands, 'YTickLabel', band_labels);
xlabel('Time (s)');
ylabel('Frequency Band');
title('EMD: Energy Distribution Across Frequency Bands');

% STFT heatmap
subplot(1, 2, 2);
imagesc(T, 1:num_bands, band_energy_stft_normalized);
colorbar;
colormap('hot');
set(gca, 'YTick', 1:num_bands, 'YTickLabel', band_labels);
xlabel('Time (s)');
ylabel('Frequency Band');
title('STFT: Energy Distribution Across Frequency Bands');

%% 12. Statistical Summary
fprintf('\n========== Statistical Summary ==========\n');
fprintf('Signal duration after skipping: %.2f seconds\n', length(ppg_filtered)/fs);
fprintf('Number of frequency bands: %d\n', num_bands);
fprintf('Number of IMFs extracted: %d\n', num_imfs);

fprintf('\n===== EMD-based Energy Statistics =====\n');
fprintf('%-15s | %-12s | %-12s | %-12s\n', 'Band', 'Mean Energy', 'Std Dev', 'Max Energy');
fprintf('--------------------------------------------------------\n');

for i = 1:num_bands
    fprintf('%-15s | %12.2e | %12.2e | %12.2e\n', ...
            band_labels{i}, mean_energy_emd(i), std_energy_emd(i), max(band_energy_emd(i, :)));
end

% Find dominant frequency band for EMD
[~, dominant_band_idx_emd] = max(mean_energy_emd);
fprintf('\nEMD Dominant frequency band: %s\n', band_labels{dominant_band_idx_emd});

fprintf('\n===== STFT-based Energy Statistics =====\n');
fprintf('%-15s | %-12s | %-12s | %-12s\n', 'Band', 'Mean Energy', 'Std Dev', 'Max Energy');
fprintf('--------------------------------------------------------\n');

for i = 1:num_bands
    fprintf('%-15s | %12.2e | %12.2e | %12.2e\n', ...
            band_labels{i}, mean_energy_stft(i), std_energy_stft(i), max(band_energy_stft(i, :)));
end

% Find dominant frequency band for STFT
[~, dominant_band_idx_stft] = max(mean_energy_stft);
fprintf('\nSTFT Dominant frequency band: %s\n', band_labels{dominant_band_idx_stft});

%% 13. Save results
fprintf('\nSaving analysis results...\n');

% Save processed data
results.filtered_signal = ppg_filtered;
results.time = t;
results.frequency_bands = freq_bands;
results.band_labels = band_labels;

% EMD results
results.emd.imfs = imfs;
results.emd.residual = residual;
results.emd.band_energy = band_energy_emd;
results.emd.band_energy_normalized = band_energy_emd_normalized;
results.emd.mean_energy = mean_energy_emd;
results.emd.std_energy = std_energy_emd;
results.emd.time = T_emd;

% STFT results
results.stft.band_energy = band_energy_stft;
results.stft.band_energy_normalized = band_energy_stft_normalized;
results.stft.mean_energy = mean_energy_stft;
results.stft.std_energy = std_energy_stft;
results.stft.spectrogram_time = T;
results.stft.spectrogram_freq = F_roi;
results.stft.spectrogram_power = P_roi;

save('ppg_analysis_results.mat', 'results');
fprintf('Results saved to ppg_analysis_results.mat\n');

% Export EMD energy data to CSV
emd_energy_table = array2table([T_emd', band_energy_emd'], ...
    'VariableNames', ['Time', band_labels]);
writetable(emd_energy_table, 'emd_band_energy_results.csv');
fprintf('EMD band energy data saved to emd_band_energy_results.csv\n');

% Export STFT energy data to CSV
stft_energy_table = array2table([T', band_energy_stft'], ...
    'VariableNames', ['Time', band_labels]);
writetable(stft_energy_table, 'stft_band_energy_results.csv');
fprintf('STFT band energy data saved to stft_band_energy_results.csv\n');

fprintf('\nAnalysis complete!\n');