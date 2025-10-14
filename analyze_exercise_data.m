%% Exercise Data Analysis and HTML Report Generator
% This script analyzes AAALogging_*.csv files containing exercise data
% and generates a comprehensive HTML report with results.
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
    
    %% 3. Generate HTML report
    fprintf('Step 3: Generating HTML report...\n');
    generate_html_report(results);
    
    fprintf('\n=== Analysis Complete ===\n');
    fprintf('HTML report saved as: exercise_analysis_report.html\n');
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

%% Function to generate HTML report
function generate_html_report(results)
    % Create HTML report file
    html_file = 'exercise_analysis_report.html';
    fid = fopen(html_file, 'w');
    
    % Write HTML header
    fprintf(fid, '<!DOCTYPE html>\n');
    fprintf(fid, '<html lang="ko">\n');
    fprintf(fid, '<head>\n');
    fprintf(fid, '    <meta charset="UTF-8">\n');
    fprintf(fid, '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n');
    fprintf(fid, '    <title>Exercise Data Analysis Report</title>\n');
    fprintf(fid, '    <style>\n');
    fprintf(fid, '        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }\n');
    fprintf(fid, '        .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }\n');
    fprintf(fid, '        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }\n');
    fprintf(fid, '        h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }\n');
    fprintf(fid, '        .summary { background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }\n');
    fprintf(fid, '        .analysis-section { margin: 30px 0; padding: 20px; border: 1px solid #bdc3c7; border-radius: 5px; }\n');
    fprintf(fid, '        .metrics { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }\n');
    fprintf(fid, '        .metric-item { background-color: #f8f9fa; padding: 10px; border-left: 3px solid #3498db; }\n');
    fprintf(fid, '        .metric-label { font-weight: bold; color: #555; }\n');
    fprintf(fid, '        .metric-value { color: #2c3e50; }\n');
    fprintf(fid, '        .intensity-high { color: #e74c3c; font-weight: bold; }\n');
    fprintf(fid, '        .intensity-moderate { color: #f39c12; font-weight: bold; }\n');
    fprintf(fid, '        .intensity-low { color: #27ae60; font-weight: bold; }\n');
    fprintf(fid, '        img { max-width: 100%%; height: auto; margin: 10px 0; border: 1px solid #ddd; }\n');
    fprintf(fid, '        table { width: 100%%; border-collapse: collapse; margin: 20px 0; }\n');
    fprintf(fid, '        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }\n');
    fprintf(fid, '        th { background-color: #3498db; color: white; }\n');
    fprintf(fid, '        tr:hover { background-color: #f5f5f5; }\n');
    fprintf(fid, '        .page-break { page-break-after: always; }\n');
    fprintf(fid, '        @media print { .page-break { page-break-after: always; } }\n');
    fprintf(fid, '    </style>\n');
    fprintf(fid, '</head>\n');
    fprintf(fid, '<body>\n');
    fprintf(fid, '    <div class="container">\n');
    
    % Add summary section
    add_html_summary(fid, results);
    
    % Add individual analysis sections
    for i = 1:length(results)
        if results{i}.success
            add_html_analysis(fid, results{i}, i);
        end
    end
    
    % Close HTML
    fprintf(fid, '    </div>\n');
    fprintf(fid, '</body>\n');
    fprintf(fid, '</html>\n');
    
    fclose(fid);
    fprintf('HTML report saved: %s\n', html_file);
end

%% Function to add HTML summary section
function add_html_summary(fid, results)
    % Add title
    fprintf(fid, '        <h1>Exercise Data Analysis - Summary</h1>\n');
    fprintf(fid, '        <p>Generated on: %s</p>\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
    
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
    
    % Create summary section
    fprintf(fid, '        <div class="summary">\n');
    fprintf(fid, '            <h2>Summary Statistics</h2>\n');
    fprintf(fid, '            <table>\n');
    fprintf(fid, '                <tr><th>Metric</th><th>Value</th></tr>\n');
    fprintf(fid, '                <tr><td>Total Files Analyzed</td><td>%d</td></tr>\n', num_files);
    fprintf(fid, '                <tr><td>Successfully Processed</td><td>%d</td></tr>\n', num_success);
    fprintf(fid, '                <tr><td>Failed</td><td>%d</td></tr>\n', num_failed);
    fprintf(fid, '                <tr><td>Average RMS (Total)</td><td>%.4f</td></tr>\n', avg_rms);
    fprintf(fid, '                <tr><td>Average SMA</td><td>%.4f</td></tr>\n', avg_sma);
    fprintf(fid, '                <tr><td>Average Magnitude</td><td>%.4f</td></tr>\n', avg_magnitude_mean);
    fprintf(fid, '            </table>\n');
    fprintf(fid, '            <h3>Intensity Distribution</h3>\n');
    fprintf(fid, '            <table>\n');
    fprintf(fid, '                <tr><th>Intensity Level</th><th>Number of Files</th></tr>\n');
    fprintf(fid, '                <tr><td class="intensity-low">Low Intensity</td><td>%d</td></tr>\n', intensity_counts.Low);
    fprintf(fid, '                <tr><td class="intensity-moderate">Moderate Intensity</td><td>%d</td></tr>\n', intensity_counts.Moderate);
    fprintf(fid, '                <tr><td class="intensity-high">High Intensity</td><td>%d</td></tr>\n', intensity_counts.High);
    fprintf(fid, '            </table>\n');
    fprintf(fid, '        </div>\n');
    fprintf(fid, '        <div class="page-break"></div>\n');
end

%% Function to add individual HTML analysis section
function add_html_analysis(fid, result, index)
    % Prepare metrics content
    m = result.metrics;
    
    % Determine intensity class
    intensity_class = 'intensity-low';
    if strcmp(m.intensity_level, 'Moderate')
        intensity_class = 'intensity-moderate';
    elseif strcmp(m.intensity_level, 'High')
        intensity_class = 'intensity-high';
    end
    
    % Add analysis section
    fprintf(fid, '        <div class="analysis-section">\n');
    fprintf(fid, '            <h2>Analysis Results - %s</h2>\n', result.file_name);
    fprintf(fid, '            <p><strong>File:</strong> %s</p>\n', result.file_name);
    fprintf(fid, '            <p><strong>Samples:</strong> %d</p>\n', result.num_samples);
    fprintf(fid, '            <p><strong>Intensity Level:</strong> <span class="%s">%s</span></p>\n', intensity_class, m.intensity_level);
    
    % Key Metrics
    fprintf(fid, '            <h3>Key Metrics</h3>\n');
    fprintf(fid, '            <div class="metrics">\n');
    fprintf(fid, '                <div class="metric-item">\n');
    fprintf(fid, '                    <span class="metric-label">Magnitude Mean:</span> <span class="metric-value">%.4f</span>\n', m.magnitude_mean);
    fprintf(fid, '                </div>\n');
    fprintf(fid, '                <div class="metric-item">\n');
    fprintf(fid, '                    <span class="metric-label">Magnitude Std:</span> <span class="metric-value">%.4f</span>\n', m.magnitude_std);
    fprintf(fid, '                </div>\n');
    fprintf(fid, '                <div class="metric-item">\n');
    fprintf(fid, '                    <span class="metric-label">Magnitude Max:</span> <span class="metric-value">%.4f</span>\n', m.magnitude_max);
    fprintf(fid, '                </div>\n');
    fprintf(fid, '                <div class="metric-item">\n');
    fprintf(fid, '                    <span class="metric-label">RMS Total:</span> <span class="metric-value">%.4f</span>\n', m.rms_total);
    fprintf(fid, '                </div>\n');
    fprintf(fid, '                <div class="metric-item">\n');
    fprintf(fid, '                    <span class="metric-label">SMA:</span> <span class="metric-value">%.4f</span>\n', m.sma);
    fprintf(fid, '                </div>\n');
    fprintf(fid, '                <div class="metric-item">\n');
    fprintf(fid, '                    <span class="metric-label">Energy Total:</span> <span class="metric-value">%.2f</span>\n', m.energy_total);
    fprintf(fid, '                </div>\n');
    fprintf(fid, '                <div class="metric-item">\n');
    fprintf(fid, '                    <span class="metric-label">Movement Variability:</span> <span class="metric-value">%.2f%%</span>\n', m.movement_variability);
    fprintf(fid, '                </div>\n');
    fprintf(fid, '            </div>\n');
    
    % RMS per axis
    fprintf(fid, '            <h3>RMS per Axis</h3>\n');
    fprintf(fid, '            <table>\n');
    fprintf(fid, '                <tr><th>Axis</th><th>RMS</th></tr>\n');
    fprintf(fid, '                <tr><td>X</td><td>%.4f</td></tr>\n', m.rms_x);
    fprintf(fid, '                <tr><td>Y</td><td>%.4f</td></tr>\n', m.rms_y);
    fprintf(fid, '                <tr><td>Z</td><td>%.4f</td></tr>\n', m.rms_z);
    fprintf(fid, '            </table>\n');
    
    % Axis Statistics
    fprintf(fid, '            <h3>Axis Statistics</h3>\n');
    fprintf(fid, '            <table>\n');
    fprintf(fid, '                <tr><th>Metric</th><th>X</th><th>Y</th><th>Z</th></tr>\n');
    fprintf(fid, '                <tr><td>Variance</td><td>%.4f</td><td>%.4f</td><td>%.4f</td></tr>\n', m.var_x, m.var_y, m.var_z);
    fprintf(fid, '                <tr><td>Peak-to-Peak</td><td>%.4f</td><td>%.4f</td><td>%.4f</td></tr>\n', m.p2p_x, m.p2p_y, m.p2p_z);
    fprintf(fid, '                <tr><td>Zero Crossing Rate</td><td>%.4f</td><td>%.4f</td><td>%.4f</td></tr>\n', m.zcr_x, m.zcr_y, m.zcr_z);
    fprintf(fid, '            </table>\n');
    
    % Correlations
    fprintf(fid, '            <h3>Correlations</h3>\n');
    fprintf(fid, '            <table>\n');
    fprintf(fid, '                <tr><th>Axes</th><th>Correlation</th></tr>\n');
    fprintf(fid, '                <tr><td>X-Y</td><td>%.4f</td></tr>\n', m.corr_xy);
    fprintf(fid, '                <tr><td>X-Z</td><td>%.4f</td></tr>\n', m.corr_xz);
    fprintf(fid, '                <tr><td>Y-Z</td><td>%.4f</td></tr>\n', m.corr_yz);
    fprintf(fid, '            </table>\n');
    
    % Add spectrogram images
    fprintf(fid, '            <h3>Spectrograms</h3>\n');
    if isfield(result.spectrograms, 'image_path') && exist(result.spectrograms.image_path, 'file')
        fprintf(fid, '            <img src="%s" alt="Spectrogram - %s">\n', result.spectrograms.image_path, result.file_name);
    end
    if isfield(result.spectrograms, 'magnitude_image_path') && exist(result.spectrograms.magnitude_image_path, 'file')
        fprintf(fid, '            <img src="%s" alt="Magnitude Spectrogram - %s">\n', result.spectrograms.magnitude_image_path, result.file_name);
    end
    
    fprintf(fid, '        </div>\n');
    fprintf(fid, '        <div class="page-break"></div>\n');
end
