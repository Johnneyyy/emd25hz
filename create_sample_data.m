%% Create Sample Exercise Data
% This script creates sample AAALogging_*.csv files for testing
% the exercise data analysis script

function create_sample_data()
    fprintf('Creating sample exercise data files...\n');
    
    % Create a sample data directory
    if ~exist('sample_data', 'dir')
        mkdir('sample_data');
    end
    
    % Create 3 sample CSV files with different exercise intensities
    create_sample_file('sample_data/AAALogging_walking.csv', 'walking');
    create_sample_file('sample_data/AAALogging_jogging.csv', 'jogging');
    create_sample_file('sample_data/AAALogging_running.csv', 'running');
    
    fprintf('Sample data created successfully!\n');
    fprintf('Files created in: %s\n', fullfile(pwd, 'sample_data'));
end

function create_sample_file(filename, activity_type)
    % Sampling parameters
    fs = 100; % 100 Hz sampling rate
    duration = 30; % 30 seconds of data
    t = (0:1/fs:duration-1/fs)';
    num_samples = length(t);
    
    % Generate synthetic acceleration data based on activity type
    switch activity_type
        case 'walking'
            % Low intensity - walking pattern
            freq = 2; % 2 Hz step frequency
            amplitude = 0.3;
            noise_level = 0.05;
            
        case 'jogging'
            % Moderate intensity - jogging pattern
            freq = 3; % 3 Hz step frequency
            amplitude = 0.8;
            noise_level = 0.1;
            
        case 'running'
            % High intensity - running pattern
            freq = 4; % 4 Hz step frequency
            amplitude = 1.5;
            noise_level = 0.15;
            
        otherwise
            freq = 2;
            amplitude = 0.5;
            noise_level = 0.1;
    end
    
    % Generate acceleration patterns
    % Vertical (Z-axis) has the most pronounced periodic pattern
    Acc_Z = amplitude * sin(2*pi*freq*t) + 9.81 + noise_level*randn(num_samples, 1);
    
    % X and Y axes have smaller amplitude and different phases
    Acc_X = 0.3*amplitude * sin(2*pi*freq*t + pi/4) + noise_level*randn(num_samples, 1);
    Acc_Y = 0.3*amplitude * sin(2*pi*freq*t + pi/2) + noise_level*randn(num_samples, 1);
    
    % Add some higher frequency components for realism
    Acc_X = Acc_X + 0.05*amplitude * sin(2*pi*freq*5*t);
    Acc_Y = Acc_Y + 0.05*amplitude * sin(2*pi*freq*6*t);
    Acc_Z = Acc_Z + 0.1*amplitude * sin(2*pi*freq*4*t);
    
    % Create table
    data = table(Acc_X, Acc_Y, Acc_Z);
    
    % Add timestamp column (optional)
    timestamp = (0:num_samples-1)';
    data = [table(timestamp), data];
    
    % Write to CSV
    writetable(data, filename);
    
    fprintf('  Created: %s (%d samples, %s activity)\n', filename, num_samples, activity_type);
end
