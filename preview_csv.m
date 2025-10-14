%% Preview Exercise CSV File
% Quick visualization tool to preview a single CSV file
% Usage: preview_csv('path/to/AAALogging_file.csv')

function preview_csv(csv_file)
    if nargin < 1
        % If no file specified, search for CSV files
        files = dir('**/AAALogging_*.csv');
        if isempty(files)
            error('No AAALogging_*.csv files found. Please specify a file path.');
        end
        csv_file = fullfile(files(1).folder, files(1).name);
        fprintf('Using first found file: %s\n', csv_file);
    end
    
    % Read the CSV file
    fprintf('Reading file: %s\n', csv_file);
    data = readtable(csv_file);
    
    % Display basic info
    fprintf('\nFile Information:\n');
    fprintf('  Number of rows: %d\n', height(data));
    fprintf('  Number of columns: %d\n', width(data));
    fprintf('  Column names: %s\n', strjoin(data.Properties.VariableNames, ', '));
    
    % Check for required columns
    if ~all(ismember({'Acc_X', 'Acc_Y', 'Acc_Z'}, data.Properties.VariableNames))
        warning('File does not contain required columns (Acc_X, Acc_Y, Acc_Z)');
        fprintf('\nAvailable columns:\n');
        disp(data.Properties.VariableNames);
        return;
    end
    
    % Extract acceleration data
    acc_x = data.Acc_X;
    acc_y = data.Acc_Y;
    acc_z = data.Acc_Z;
    
    % Calculate magnitude
    acc_mag = sqrt(acc_x.^2 + acc_y.^2 + acc_z.^2);
    
    % Display statistics
    fprintf('\nAcceleration Statistics:\n');
    fprintf('  Acc_X: mean=%.4f, std=%.4f, range=[%.4f, %.4f]\n', ...
        mean(acc_x), std(acc_x), min(acc_x), max(acc_x));
    fprintf('  Acc_Y: mean=%.4f, std=%.4f, range=[%.4f, %.4f]\n', ...
        mean(acc_y), std(acc_y), min(acc_y), max(acc_y));
    fprintf('  Acc_Z: mean=%.4f, std=%.4f, range=[%.4f, %.4f]\n', ...
        mean(acc_z), std(acc_z), min(acc_z), max(acc_z));
    fprintf('  Magnitude: mean=%.4f, std=%.4f, range=[%.4f, %.4f]\n', ...
        mean(acc_mag), std(acc_mag), min(acc_mag), max(acc_mag));
    
    % Create visualization
    figure('Position', [100, 100, 1200, 800]);
    
    % Subplot 1: Time series of all three axes
    subplot(3, 2, [1, 2]);
    t = (0:length(acc_x)-1)';
    plot(t, acc_x, 'r', 'LineWidth', 1); hold on;
    plot(t, acc_y, 'g', 'LineWidth', 1);
    plot(t, acc_z, 'b', 'LineWidth', 1);
    plot(t, acc_mag, 'k', 'LineWidth', 1.5);
    legend('Acc_X', 'Acc_Y', 'Acc_Z', 'Magnitude', 'Location', 'best');
    xlabel('Sample Number');
    ylabel('Acceleration');
    title(sprintf('Acceleration Time Series - %s', strrep(csv_file, '_', '\_')));
    grid on;
    hold off;
    
    % Subplot 2: Histogram of magnitude
    subplot(3, 2, 3);
    histogram(acc_mag, 50, 'FaceColor', 'blue', 'EdgeColor', 'black');
    xlabel('Magnitude');
    ylabel('Frequency');
    title('Magnitude Distribution');
    grid on;
    
    % Subplot 3: 3D trajectory (first 1000 points)
    subplot(3, 2, 4);
    n_points = min(1000, length(acc_x));
    plot3(acc_x(1:n_points), acc_y(1:n_points), acc_z(1:n_points), 'b-', 'LineWidth', 0.5);
    xlabel('Acc_X');
    ylabel('Acc_Y');
    zlabel('Acc_Z');
    title(sprintf('3D Trajectory (first %d points)', n_points));
    grid on;
    rotate3d on;
    
    % Subplot 4: Power Spectral Density
    subplot(3, 2, 5);
    fs = 100; % Assumed sampling frequency
    [pxx, f] = pwelch(acc_mag, [], [], [], fs);
    plot(f, 10*log10(pxx), 'LineWidth', 1.5);
    xlabel('Frequency (Hz)');
    ylabel('Power/Frequency (dB/Hz)');
    title('Power Spectral Density');
    grid on;
    xlim([0, 20]); % Focus on 0-20 Hz
    
    % Subplot 5: Autocorrelation
    subplot(3, 2, 6);
    [acf, lags] = xcorr(acc_mag, min(500, floor(length(acc_mag)/4)), 'normalized');
    plot(lags, acf, 'LineWidth', 1.5);
    xlabel('Lag');
    ylabel('Autocorrelation');
    title('Autocorrelation Function');
    grid on;
    
    fprintf('\nVisualization complete!\n');
end
