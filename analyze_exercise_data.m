%% Exercise Data Analysis and PowerPoint Report Generator
% This script analyzes AAALogging_*.csv files containing exercise data
% and generates a comprehensive PowerPoint presentation with results.
%
% Requirements:
% - MATLAB R2016b or later
% - Statistics and Machine Learning Toolbox (optional, for advanced metrics)
% - Signal Processing Toolbox (for spectrogram)
%
% Author: Auto-generated
% Date: 2025-10-14

function analyze_exercise_data()
    %% Initialize
    close all;
    clc;
    
    fprintf('=== Exercise Data Analysis Starting ===\n\n');
    
    %% 1. Find all AAALogging_*.csv files
    fprintf('Step 1: Searching for AAALogging_*.csv files...\n');
    csv_files = find_exercise_files();
    
    if isempty(csv_files)
        error('No AAALogging_*.csv files found in the current directory or subdirectories.');
    end
    
    fprintf('Found %d exercise CSV files.\n\n', length(csv_files));
    
    %% 2. Analyze each file
    fprintf('Step 2: Analyzing exercise data...\n');
    results = cell(length(csv_files), 1);
    
    for i = 1:length(csv_files)
        fprintf('  Processing file %d/%d: %s\n', i, length(csv_files), csv_files{i});
        results{i} = analyze_single_file(csv_files{i});
    end
    
    fprintf('\nAnalysis complete!\n\n');
    
    %% 3. Generate PowerPoint presentation
    fprintf('Step 3: Generating PowerPoint presentation...\n');
    generate_powerpoint(results);
    
    fprintf('\n=== Analysis Complete ===\n');
    fprintf('PowerPoint file saved as: exercise_analysis_report.pptx\n');
end

%% Function to find all AAALogging_*.csv files recursively
function csv_files = find_exercise_files()
    % Get current directory
    root_dir = pwd;
    
    % Find all CSV files matching the pattern
    csv_files = {};
    search_directory(root_dir);
    
    function search_directory(dir_path)
        % Get all items in directory
        items = dir(dir_path);
        
        for j = 1:length(items)
            item = items(j);
            
            % Skip . and ..
            if strcmp(item.name, '.') || strcmp(item.name, '..')
                continue;
            end
            
            full_path = fullfile(dir_path, item.name);
            
            if item.isdir
                % Recursively search subdirectories
                search_directory(full_path);
            else
                % Check if file matches pattern
                if ~isempty(regexp(item.name, '^AAALogging_.*\.csv$', 'once'))
                    csv_files{end+1} = full_path;
                end
            end
        end
    end
end

%% Function to analyze a single CSV file
function result = analyze_single_file(file_path)
    result = struct();
    result.file_path = file_path;
    [~, result.file_name, ~] = fileparts(file_path);
    
    try
        % Read CSV file
        data = readtable(file_path);
        
        % Check if required columns exist
        if ~all(ismember({'Acc_X', 'Acc_Y', 'Acc_Z'}, data.Properties.VariableNames))
            warning('File %s does not contain required columns (Acc_X, Acc_Y, Acc_Z)', file_path);
            result.error = 'Missing required columns';
            return;
        end
        
        % Extract acceleration data
        acc_x = data.Acc_X;
        acc_y = data.Acc_Y;
        acc_z = data.Acc_Z;
        
        % Store raw data
        result.acc_x = acc_x;
        result.acc_y = acc_y;
        result.acc_z = acc_z;
        result.num_samples = length(acc_x);
        
        % Calculate exercise intensity metrics
        result.metrics = calculate_intensity_metrics(acc_x, acc_y, acc_z);
        
        % Generate spectrograms
        result.spectrograms = generate_spectrograms(acc_x, acc_y, acc_z, result.file_name);
        
        result.success = true;
        
    catch ME
        warning('Error processing file %s: %s', file_path, ME.message);
        result.error = ME.message;
        result.success = false;
    end
end

%% Function to calculate various exercise intensity metrics
function metrics = calculate_intensity_metrics(acc_x, acc_y, acc_z)
    metrics = struct();
    
    % 1. Magnitude of acceleration vector
    acc_magnitude = sqrt(acc_x.^2 + acc_y.^2 + acc_z.^2);
    metrics.magnitude_mean = mean(acc_magnitude);
    metrics.magnitude_std = std(acc_magnitude);
    metrics.magnitude_max = max(acc_magnitude);
    metrics.magnitude_min = min(acc_magnitude);
    
    % 2. Signal Magnitude Area (SMA) - commonly used in activity recognition
    metrics.sma = (sum(abs(acc_x)) + sum(abs(acc_y)) + sum(abs(acc_z))) / length(acc_x);
    
    % 3. Root Mean Square (RMS) for each axis
    metrics.rms_x = sqrt(mean(acc_x.^2));
    metrics.rms_y = sqrt(mean(acc_y.^2));
    metrics.rms_z = sqrt(mean(acc_z.^2));
    metrics.rms_total = sqrt(mean(acc_magnitude.^2));
    
    % 4. Peak-to-Peak amplitude
    metrics.p2p_x = max(acc_x) - min(acc_x);
    metrics.p2p_y = max(acc_y) - min(acc_y);
    metrics.p2p_z = max(acc_z) - min(acc_z);
    
    % 5. Energy (sum of squared values)
    metrics.energy_x = sum(acc_x.^2);
    metrics.energy_y = sum(acc_y.^2);
    metrics.energy_z = sum(acc_z.^2);
    metrics.energy_total = sum(acc_magnitude.^2);
    
    % 6. Zero Crossing Rate (ZCR) - indicates frequency of direction changes
    metrics.zcr_x = sum(abs(diff(sign(acc_x)))) / (2 * length(acc_x));
    metrics.zcr_y = sum(abs(diff(sign(acc_y)))) / (2 * length(acc_y));
    metrics.zcr_z = sum(abs(diff(sign(acc_z)))) / (2 * length(acc_z));
    
    % 7. Variance for each axis
    metrics.var_x = var(acc_x);
    metrics.var_y = var(acc_y);
    metrics.var_z = var(acc_z);
    
    % 8. Interquartile Range (IQR)
    metrics.iqr_x = iqr(acc_x);
    metrics.iqr_y = iqr(acc_y);
    metrics.iqr_z = iqr(acc_z);
    
    % 9. Skewness and Kurtosis
    metrics.skewness_x = skewness(acc_x);
    metrics.skewness_y = skewness(acc_y);
    metrics.skewness_z = skewness(acc_z);
    metrics.kurtosis_x = kurtosis(acc_x);
    metrics.kurtosis_y = kurtosis(acc_y);
    metrics.kurtosis_z = kurtosis(acc_z);
    
    % 10. Correlation between axes
    metrics.corr_xy = corr(acc_x, acc_y);
    metrics.corr_xz = corr(acc_x, acc_z);
    metrics.corr_yz = corr(acc_y, acc_z);
    
    % 11. Intensity Level Classification (based on magnitude)
    % Simple classification based on RMS of magnitude
    if metrics.rms_total < 0.5
        metrics.intensity_level = 'Low';
    elseif metrics.rms_total < 1.5
        metrics.intensity_level = 'Moderate';
    else
        metrics.intensity_level = 'High';
    end
    
    % 12. Movement Variability Index (coefficient of variation of magnitude)
    metrics.movement_variability = (metrics.magnitude_std / metrics.magnitude_mean) * 100;
end

%% Function to generate spectrograms
function spectrograms = generate_spectrograms(acc_x, acc_y, acc_z, file_name)
    spectrograms = struct();
    
    % Sampling frequency (assume 100 Hz if not specified, adjust as needed)
    fs = 100; % Hz
    
    % Create figure for spectrograms
    fig = figure('Visible', 'off', 'Position', [100, 100, 1200, 800]);
    
    % Spectrogram parameters
    window_length = min(256, floor(length(acc_x)/4));
    overlap = floor(window_length * 0.75);
    nfft = max(256, 2^nextpow2(window_length));
    
    % Subplot 1: Acc_X spectrogram
    subplot(3, 1, 1);
    spectrogram(acc_x, window_length, overlap, nfft, fs, 'yaxis');
    title(sprintf('Spectrogram - Acc_X (%s)', strrep(file_name, '_', '\_')));
    colorbar;
    
    % Subplot 2: Acc_Y spectrogram
    subplot(3, 1, 2);
    spectrogram(acc_y, window_length, overlap, nfft, fs, 'yaxis');
    title(sprintf('Spectrogram - Acc_Y (%s)', strrep(file_name, '_', '\_')));
    colorbar;
    
    % Subplot 3: Acc_Z spectrogram
    subplot(3, 1, 3);
    spectrogram(acc_z, window_length, overlap, nfft, fs, 'yaxis');
    title(sprintf('Spectrogram - Acc_Z (%s)', strrep(file_name, '_', '\_')));
    colorbar;
    
    % Save figure
    spectrograms.figure_handle = fig;
    spectrograms.image_path = sprintf('spectrogram_%s.png', file_name);
    saveas(fig, spectrograms.image_path);
    
    % Also create a combined magnitude spectrogram
    acc_magnitude = sqrt(acc_x.^2 + acc_y.^2 + acc_z.^2);
    fig_mag = figure('Visible', 'off', 'Position', [100, 100, 800, 600]);
    spectrogram(acc_magnitude, window_length, overlap, nfft, fs, 'yaxis');
    title(sprintf('Spectrogram - Acceleration Magnitude (%s)', strrep(file_name, '_', '\_')));
    colorbar;
    spectrograms.magnitude_figure = fig_mag;
    spectrograms.magnitude_image_path = sprintf('spectrogram_magnitude_%s.png', file_name);
    saveas(fig_mag, spectrograms.magnitude_image_path);
end

%% Function to generate PowerPoint presentation
function generate_powerpoint(results)
    % Import required Java classes for PowerPoint generation
    import mlreportgen.ppt.*;
    
    % Create presentation
    ppt_file = 'exercise_analysis_report.pptx';
    ppt = Presentation(ppt_file);
    open(ppt);
    
    % Slide 1: Summary Page
    add_summary_slide(ppt, results);
    
    % Slides 2-N: Individual analysis pages
    for i = 1:length(results)
        if results{i}.success
            add_analysis_slide(ppt, results{i}, i);
        end
    end
    
    % Close and save presentation
    close(ppt);
    
    fprintf('PowerPoint presentation saved: %s\n', ppt_file);
end

%% Function to add summary slide
function add_summary_slide(ppt, results)
    slide = add(ppt, 'Title and Content');
    
    % Add title
    replace(slide, 'Title', 'Exercise Data Analysis - Summary');
    
    % Prepare summary content
    num_files = length(results);
    num_success = sum(cellfun(@(r) isfield(r, 'success') && r.success, results));
    num_failed = num_files - num_success;
    
    % Calculate overall statistics
    all_metrics = {};
    for i = 1:length(results)
        if results{i}.success
            all_metrics{end+1} = results{i}.metrics;
        end
    end
    
    if ~isempty(all_metrics)
        avg_rms = mean(cellfun(@(m) m.rms_total, all_metrics));
        avg_sma = mean(cellfun(@(m) m.sma, all_metrics));
        avg_magnitude_mean = mean(cellfun(@(m) m.magnitude_mean, all_metrics));
        
        intensity_counts = struct('Low', 0, 'Moderate', 0, 'High', 0);
        for i = 1:length(all_metrics)
            level = all_metrics{i}.intensity_level;
            intensity_counts.(level) = intensity_counts.(level) + 1;
        end
    else
        avg_rms = 0;
        avg_sma = 0;
        avg_magnitude_mean = 0;
        intensity_counts = struct('Low', 0, 'Moderate', 0, 'High', 0);
    end
    
    % Create summary text
    summary_text = sprintf(['Analysis Summary\n\n' ...
                           'Total Files Analyzed: %d\n' ...
                           'Successfully Processed: %d\n' ...
                           'Failed: %d\n\n' ...
                           'Overall Statistics:\n' ...
                           '  Average RMS (Total): %.4f\n' ...
                           '  Average SMA: %.4f\n' ...
                           '  Average Magnitude: %.4f\n\n' ...
                           'Intensity Distribution:\n' ...
                           '  Low Intensity: %d files\n' ...
                           '  Moderate Intensity: %d files\n' ...
                           '  High Intensity: %d files\n'], ...
                           num_files, num_success, num_failed, ...
                           avg_rms, avg_sma, avg_magnitude_mean, ...
                           intensity_counts.Low, intensity_counts.Moderate, intensity_counts.High);
    
    % Add content
    replace(slide, 'Content', summary_text);
end

%% Function to add individual analysis slide
function add_analysis_slide(ppt, result, index)
    slide = add(ppt, 'Title and Content');
    
    % Add title
    title_text = sprintf('Analysis Results - %s', result.file_name);
    replace(slide, 'Title', title_text);
    
    % Prepare metrics content
    m = result.metrics;
    
    content_text = sprintf(['File: %s\n' ...
                           'Samples: %d\n\n' ...
                           'Intensity Level: %s\n\n' ...
                           'Key Metrics:\n' ...
                           '  Magnitude: Mean=%.4f, Std=%.4f, Max=%.4f\n' ...
                           '  RMS: X=%.4f, Y=%.4f, Z=%.4f, Total=%.4f\n' ...
                           '  SMA: %.4f\n' ...
                           '  Energy Total: %.2f\n' ...
                           '  Movement Variability: %.2f%%\n\n' ...
                           'Axis Statistics:\n' ...
                           '  Variance: X=%.4f, Y=%.4f, Z=%.4f\n' ...
                           '  Peak-to-Peak: X=%.4f, Y=%.4f, Z=%.4f\n' ...
                           '  Zero Crossing Rate: X=%.4f, Y=%.4f, Z=%.4f\n\n' ...
                           'Correlations:\n' ...
                           '  X-Y: %.4f, X-Z: %.4f, Y-Z: %.4f\n'], ...
                           result.file_name, result.num_samples, m.intensity_level, ...
                           m.magnitude_mean, m.magnitude_std, m.magnitude_max, ...
                           m.rms_x, m.rms_y, m.rms_z, m.rms_total, ...
                           m.sma, m.energy_total, m.movement_variability, ...
                           m.var_x, m.var_y, m.var_z, ...
                           m.p2p_x, m.p2p_y, m.p2p_z, ...
                           m.zcr_x, m.zcr_y, m.zcr_z, ...
                           m.corr_xy, m.corr_xz, m.corr_yz);
    
    replace(slide, 'Content', content_text);
    
    % Add spectrogram slide
    slide_spec = add(ppt, 'Title and Content');
    replace(slide_spec, 'Title', sprintf('Spectrograms - %s', result.file_name));
    
    % Add spectrogram images
    if isfield(result.spectrograms, 'image_path') && exist(result.spectrograms.image_path, 'file')
        pic = Picture(result.spectrograms.image_path);
        pic.X = '1in';
        pic.Y = '1.5in';
        pic.Width = '8in';
        pic.Height = '5in';
        replace(slide_spec, 'Content', pic);
    end
end
