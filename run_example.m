%% Example: Complete Exercise Data Analysis Workflow
% This script demonstrates the complete workflow:
% 1. Create sample data (if needed)
% 2. Run the analysis
% 3. Display results

clc;
clear;
close all;

fprintf('========================================\n');
fprintf('Exercise Data Analysis Example\n');
fprintf('========================================\n\n');

%% Step 1: Check if sample data exists, create if needed
if ~exist('sample_data', 'dir') || isempty(dir('sample_data/AAALogging_*.csv'))
    fprintf('Step 1: Creating sample exercise data...\n');
    create_sample_data();
    fprintf('\n');
else
    fprintf('Step 1: Sample data already exists.\n\n');
end

%% Step 2: Run the analysis
fprintf('Step 2: Running exercise data analysis...\n');
fprintf('This will analyze all AAALogging_*.csv files in subdirectories.\n\n');

% Make sure we're in the right directory
cd(fileparts(which('run_example.m')));

% Run the main analysis function
try
    analyze_exercise_data();
    fprintf('\n');
    fprintf('========================================\n');
    fprintf('Analysis completed successfully!\n');
    fprintf('========================================\n\n');
    
    fprintf('Output files:\n');
    fprintf('  - exercise_analysis_report.pptx (PowerPoint presentation)\n');
    fprintf('  - excel_exports/*.xlsx (Excel files for each exercise)\n');
    fprintf('  - spectrogram_*.png (Spectrogram images)\n\n');
    
    fprintf('Next steps:\n');
    fprintf('  1. Open exercise_analysis_report.pptx to view the PowerPoint presentation\n');
    fprintf('  2. Open excel_exports/ folder to view individual Excel analysis files\n');
    fprintf('  3. Review the spectrograms for frequency analysis\n');
    fprintf('  4. Replace sample data with your own AAALogging_*.csv files\n\n');
    
catch ME
    fprintf('\n========================================\n');
    fprintf('Error during analysis:\n');
    fprintf('%s\n', ME.message);
    fprintf('========================================\n');
    rethrow(ME);
end
