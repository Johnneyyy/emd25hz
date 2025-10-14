%% Exercise Data Analysis and PowerPoint/Excel Report Generator
% This script analyzes AAALogging_*.csv files containing exercise data
% and generates a comprehensive PowerPoint presentation and Excel exports.
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
    
    %% 3. Export to Excel
    fprintf('Step 3: Exporting results to Excel...\n');
    export_results_to_excel(results);
    
    %% 4. Generate PowerPoint report
    fprintf('\nStep 4: Generating PowerPoint presentation...\n');
    generate_ppt_report(results);
    
    fprintf('\n=== Analysis Complete ===\n');
    fprintf('PowerPoint report saved as: exercise_analysis_report.pptx\n');
    fprintf('Excel files saved in: excel_exports/\n');
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

%% Function to export results to Excel
function export_results_to_excel(results)
    % Create output directory for Excel files
    excel_dir = 'excel_exports';
    if ~exist(excel_dir, 'dir')
        mkdir(excel_dir);
    end
    
    % Export each exercise result to a separate Excel file
    for i = 1:length(results)
        if ~results{i}.success
            continue;
        end
        
        result = results{i};
        excel_file = fullfile(excel_dir, sprintf('%s_analysis.xlsx', result.file_name));
        
        try
            % Create Excel COM object
            Excel = actxserver('Excel.Application');
            Excel.Visible = 0; % Don't show Excel window
            
            % Create new workbook
            Workbooks = Excel.Workbooks;
            Workbook = Workbooks.Add();
            
            % Sheet 1: Summary Metrics
            Sheet1 = Workbook.Worksheets.Item(1);
            Sheet1.Name = 'Summary Metrics';
            add_excel_summary(Sheet1, result);
            
            % Sheet 2: Detailed Statistics
            Sheet2 = Workbook.Worksheets.Add([], Sheet1);
            Sheet2.Name = 'Detailed Stats';
            add_excel_detailed_stats(Sheet2, result);
            
            % Sheet 3: Raw Data
            Sheet3 = Workbook.Worksheets.Add([], Sheet2);
            Sheet3.Name = 'Raw Data';
            add_excel_raw_data(Sheet3, result);
            
            % Save and close
            Workbook.SaveAs(fullfile(pwd, excel_file));
            Workbook.Close(false);
            Excel.Quit();
            delete(Excel);
            
            fprintf('  Exported: %s\n', excel_file);
            
        catch ME
            warning('Error exporting to Excel: %s', ME.message);
            try
                Excel.Quit();
                delete(Excel);
            catch
            end
        end
    end
end

%% Function to add summary metrics to Excel sheet
function add_excel_summary(Sheet, result)
    m = result.metrics;
    
    % Title
    Sheet.Range('A1').Value = sprintf('Exercise Analysis: %s', result.file_name);
    Sheet.Range('A1').Font.Size = 14;
    Sheet.Range('A1').Font.Bold = true;
    
    % Basic info
    row = 3;
    Sheet.Range(sprintf('A%d', row)).Value = 'File Name:';
    Sheet.Range(sprintf('B%d', row)).Value = result.file_name;
    row = row + 1;
    Sheet.Range(sprintf('A%d', row)).Value = 'Number of Samples:';
    Sheet.Range(sprintf('B%d', row)).Value = result.num_samples;
    row = row + 1;
    Sheet.Range(sprintf('A%d', row)).Value = 'Intensity Level:';
    Sheet.Range(sprintf('B%d', row)).Value = m.intensity_level;
    
    % Key Metrics section
    row = row + 2;
    Sheet.Range(sprintf('A%d', row)).Value = 'KEY METRICS';
    Sheet.Range(sprintf('A%d', row)).Font.Bold = true;
    row = row + 1;
    
    metrics_data = {
        'Magnitude Mean', m.magnitude_mean;
        'Magnitude Std', m.magnitude_std;
        'Magnitude Max', m.magnitude_max;
        'Magnitude Min', m.magnitude_min;
        'RMS Total', m.rms_total;
        'SMA', m.sma;
        'Energy Total', m.energy_total;
        'Movement Variability (%)', m.movement_variability
    };
    
    for i = 1:size(metrics_data, 1)
        Sheet.Range(sprintf('A%d', row)).Value = metrics_data{i, 1};
        Sheet.Range(sprintf('B%d', row)).Value = metrics_data{i, 2};
        row = row + 1;
    end
    
    % Auto-fit columns
    Sheet.Columns.Item('A:B').AutoFit();
end

%% Function to add detailed statistics to Excel sheet
function add_excel_detailed_stats(Sheet, result)
    m = result.metrics;
    
    % Title
    Sheet.Range('A1').Value = 'Detailed Statistics';
    Sheet.Range('A1').Font.Size = 14;
    Sheet.Range('A1').Font.Bold = true;
    
    % RMS per axis
    row = 3;
    Sheet.Range(sprintf('A%d', row)).Value = 'RMS PER AXIS';
    Sheet.Range(sprintf('A%d', row)).Font.Bold = true;
    row = row + 1;
    Sheet.Range(sprintf('A%d:C%d', row, row)).Value = {'Axis', 'RMS', ''};
    row = row + 1;
    Sheet.Range(sprintf('A%d:C%d', row, row+2)).Value = {
        'X', m.rms_x, '';
        'Y', m.rms_y, '';
        'Z', m.rms_z, ''
    };
    
    % Variance
    row = row + 4;
    Sheet.Range(sprintf('A%d', row)).Value = 'VARIANCE';
    Sheet.Range(sprintf('A%d', row)).Font.Bold = true;
    row = row + 1;
    Sheet.Range(sprintf('A%d:C%d', row, row+2)).Value = {
        'X', m.var_x, '';
        'Y', m.var_y, '';
        'Z', m.var_z, ''
    };
    
    % Peak-to-Peak
    row = row + 4;
    Sheet.Range(sprintf('A%d', row)).Value = 'PEAK-TO-PEAK';
    Sheet.Range(sprintf('A%d', row)).Font.Bold = true;
    row = row + 1;
    Sheet.Range(sprintf('A%d:C%d', row, row+2)).Value = {
        'X', m.p2p_x, '';
        'Y', m.p2p_y, '';
        'Z', m.p2p_z, ''
    };
    
    % Zero Crossing Rate
    row = row + 4;
    Sheet.Range(sprintf('A%d', row)).Value = 'ZERO CROSSING RATE';
    Sheet.Range(sprintf('A%d', row)).Font.Bold = true;
    row = row + 1;
    Sheet.Range(sprintf('A%d:C%d', row, row+2)).Value = {
        'X', m.zcr_x, '';
        'Y', m.zcr_y, '';
        'Z', m.zcr_z, ''
    };
    
    % Correlations
    row = row + 4;
    Sheet.Range(sprintf('A%d', row)).Value = 'CORRELATIONS';
    Sheet.Range(sprintf('A%d', row)).Font.Bold = true;
    row = row + 1;
    Sheet.Range(sprintf('A%d:C%d', row, row+2)).Value = {
        'X-Y', m.corr_xy, '';
        'X-Z', m.corr_xz, '';
        'Y-Z', m.corr_yz, ''
    };
    
    % Auto-fit columns
    Sheet.Columns.Item('A:C').AutoFit();
end

%% Function to add raw data to Excel sheet
function add_excel_raw_data(Sheet, result)
    % Title
    Sheet.Range('A1').Value = 'Raw Acceleration Data';
    Sheet.Range('A1').Font.Size = 14;
    Sheet.Range('A1').Font.Bold = true;
    
    % Headers
    Sheet.Range('A3:C3').Value = {'Acc_X', 'Acc_Y', 'Acc_Z'};
    Sheet.Range('A3:C3').Font.Bold = true;
    
    % Data
    num_samples = length(result.acc_x);
    data = [result.acc_x, result.acc_y, result.acc_z];
    
    % Write data in chunks to avoid memory issues
    chunk_size = 1000;
    for i = 1:chunk_size:num_samples
        end_idx = min(i + chunk_size - 1, num_samples);
        start_row = i + 3;
        end_row = end_idx + 3;
        Sheet.Range(sprintf('A%d:C%d', start_row, end_row)).Value = data(i:end_idx, :);
    end
    
    % Auto-fit columns
    Sheet.Columns.Item('A:C').AutoFit();
end

%% Function to generate PowerPoint report
function generate_ppt_report(results)
    try
        % Create PowerPoint COM object
        PPT = actxserver('PowerPoint.Application');
        PPT.Visible = 1; % Show PowerPoint window
        
        % Create new presentation
        Presentations = PPT.Presentations;
        Presentation = Presentations.Add();
        
        % Add title slide
        add_ppt_title_slide(Presentation, results);
        
        % Add summary slide
        add_ppt_summary_slide(Presentation, results);
        
        % Add slides for each exercise result
        for i = 1:length(results)
            if results{i}.success
                add_ppt_exercise_slide(Presentation, results{i}, i);
            end
        end
        
        % Save presentation
        ppt_file = fullfile(pwd, 'exercise_analysis_report.pptx');
        Presentation.SaveAs(ppt_file);
        
        fprintf('PowerPoint report saved: %s\n', ppt_file);
        
        % Close PowerPoint
        Presentation.Close();
        PPT.Quit();
        delete(PPT);
        
    catch ME
        warning('Error creating PowerPoint presentation: %s', ME.message);
        try
            PPT.Quit();
            delete(PPT);
        catch
        end
        rethrow(ME);
    end
end

%% Function to add title slide to PowerPoint
function add_ppt_title_slide(Presentation, results)
    % Add slide with title layout (layout 1)
    Slide = Presentation.Slides.Add(1, 1);
    
    % Set title
    Title = Slide.Shapes.Item(1);
    Title.TextFrame.TextRange.Text = 'Exercise Data Analysis Report';
    Title.TextFrame.TextRange.Font.Size = 44;
    Title.TextFrame.TextRange.Font.Bold = true;
    
    % Set subtitle
    Subtitle = Slide.Shapes.Item(2);
    subtitle_text = sprintf('Generated on: %s\nTotal Exercises Analyzed: %d', ...
        datestr(now, 'yyyy-mm-dd HH:MM:SS'), ...
        length(results));
    Subtitle.TextFrame.TextRange.Text = subtitle_text;
    Subtitle.TextFrame.TextRange.Font.Size = 24;
end

%% Function to add summary slide to PowerPoint
function add_ppt_summary_slide(Presentation, results)
    % Add slide with title and content layout (layout 2)
    Slide = Presentation.Slides.Add(Presentation.Slides.Count + 1, 2);
    
    % Set title
    Title = Slide.Shapes.Item(1);
    Title.TextFrame.TextRange.Text = 'Summary Statistics';
    
    % Calculate statistics
    num_files = length(results);
    num_success = sum(cellfun(@(r) isfield(r, 'success') && r.success, results));
    num_failed = num_files - num_success;
    
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
    
    % Add content textbox
    Content = Slide.Shapes.Item(2);
    content_text = sprintf(['Overall Statistics:\n\n', ...
        '• Total Files: %d\n', ...
        '• Successfully Processed: %d\n', ...
        '• Failed: %d\n\n', ...
        'Average Metrics:\n', ...
        '• RMS Total: %.4f\n', ...
        '• SMA: %.4f\n', ...
        '• Magnitude Mean: %.4f\n\n', ...
        'Intensity Distribution:\n', ...
        '• Low: %d exercises\n', ...
        '• Moderate: %d exercises\n', ...
        '• High: %d exercises'], ...
        num_files, num_success, num_failed, ...
        avg_rms, avg_sma, avg_magnitude_mean, ...
        intensity_counts.Low, intensity_counts.Moderate, intensity_counts.High);
    
    Content.TextFrame.TextRange.Text = content_text;
    Content.TextFrame.TextRange.Font.Size = 18;
end

%% Function to add exercise analysis slide to PowerPoint
function add_ppt_exercise_slide(Presentation, result, index)
    m = result.metrics;
    
    % Add slide with title and content layout
    Slide = Presentation.Slides.Add(Presentation.Slides.Count + 1, 2);
    
    % Set title
    Title = Slide.Shapes.Item(1);
    Title.TextFrame.TextRange.Text = sprintf('Exercise Analysis: %s', result.file_name);
    
    % Add metrics content
    Content = Slide.Shapes.Item(2);
    content_text = sprintf(['File: %s\n', ...
        'Samples: %d\n', ...
        'Intensity Level: %s\n\n', ...
        'Key Metrics:\n', ...
        '• Magnitude Mean: %.4f\n', ...
        '• Magnitude Std: %.4f\n', ...
        '• RMS Total: %.4f\n', ...
        '• SMA: %.4f\n', ...
        '• Energy Total: %.2f\n\n', ...
        'RMS per Axis:\n', ...
        '• X: %.4f  Y: %.4f  Z: %.4f\n\n', ...
        'Correlations:\n', ...
        '• X-Y: %.4f  X-Z: %.4f  Y-Z: %.4f'], ...
        result.file_name, result.num_samples, m.intensity_level, ...
        m.magnitude_mean, m.magnitude_std, m.rms_total, m.sma, m.energy_total, ...
        m.rms_x, m.rms_y, m.rms_z, ...
        m.corr_xy, m.corr_xz, m.corr_yz);
    
    Content.TextFrame.TextRange.Text = content_text;
    Content.TextFrame.TextRange.Font.Size = 14;
    
    % Add spectrogram image if available
    if isfield(result.spectrograms, 'magnitude_image_path') && exist(result.spectrograms.magnitude_image_path, 'file')
        img_path = fullfile(pwd, result.spectrograms.magnitude_image_path);
        
        % Add a new slide for the spectrogram
        ImgSlide = Presentation.Slides.Add(Presentation.Slides.Count + 1, 11); % Blank layout
        
        % Add title manually
        TitleShape = ImgSlide.Shapes.AddTextbox(1, 50, 30, 600, 50);
        TitleShape.TextFrame.TextRange.Text = sprintf('Spectrogram: %s', result.file_name);
        TitleShape.TextFrame.TextRange.Font.Size = 32;
        TitleShape.TextFrame.TextRange.Font.Bold = true;
        
        % Add image
        ImgSlide.Shapes.AddPicture(img_path, 0, 1, 100, 150, 600, 400);
    end
end
