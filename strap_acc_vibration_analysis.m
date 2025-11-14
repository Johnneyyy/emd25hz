%% 워치 스트랩 가속도계 센서 미세 진동 분석
% 25Hz 샘플링 주파수의 ACC X/Y/Z 신호 분석
% Author: MATLAB Signal Processing
% Date: 2024

clear all; close all; clc;

%% 1. 파라미터 설정
fs = 25;                    % 샘플링 주파수 (Hz) - 실제 데이터에 맞게 수정 가능
duration = 60;              % 측정 시간 (초) - 실제 데이터에 맞게 자동 계산됨

fprintf('=== 워치 스트랩 ACC 센서 미세 진동 분석 ===\n');

%% 2. 데이터 로드 (실제 데이터 사용)
% 방법 1: CSV 파일에서 로드 (권장)
% CSV 파일 형식 예시:
%   - 컬럼명: Time, ACC_X, ACC_Y, ACC_Z 또는
%   - 컬럼명: timestamp, acc_x, acc_y, acc_z 또는
%   - 컬럼명: t, x, y, z
% 
% 아래의 data_file_path를 실제 데이터 파일 경로로 변경하세요

data_file_path = '';  % 예: 'data/strap_A.csv' 또는 'data/strap_B.csv'
strap_name = '';       % 예: 'Strap_A' 또는 'Strap_B' (출력 파일명에 사용)

% CSV 파일에서 데이터 로드 시도
if ~isempty(data_file_path) && exist(data_file_path, 'file')
    fprintf('CSV 파일에서 데이터 로드 중: %s\n', data_file_path);
    try
        data_table = readtable(data_file_path);
        
        % 컬럼명 매핑 (다양한 형식 지원)
        if ismember('ACC_X', data_table.Properties.VariableNames)
            acc_x = data_table.ACC_X;
            acc_y = data_table.ACC_Y;
            acc_z = data_table.ACC_Z;
        elseif ismember('acc_x', data_table.Properties.VariableNames)
            acc_x = data_table.acc_x;
            acc_y = data_table.acc_y;
            acc_z = data_table.acc_z;
        elseif ismember('X', data_table.Properties.VariableNames)
            acc_x = data_table.X;
            acc_y = data_table.Y;
            acc_z = data_table.Z;
        elseif ismember('x', data_table.Properties.VariableNames)
            acc_x = data_table.x;
            acc_y = data_table.y;
            acc_z = data_table.z;
        else
            % 첫 번째 컬럼이 시간, 다음 3개가 X, Y, Z라고 가정
            acc_x = data_table{:, 2};
            acc_y = data_table{:, 3};
            acc_z = data_table{:, 4};
        end
        
        % 시간 벡터 생성
        if ismember('Time', data_table.Properties.VariableNames)
            t = data_table.Time;
        elseif ismember('time', data_table.Properties.VariableNames)
            t = data_table.time;
        elseif ismember('timestamp', data_table.Properties.VariableNames)
            t = data_table.timestamp;
        elseif ismember('t', data_table.Properties.VariableNames)
            t = data_table.t;
        else
            % 시간 정보가 없으면 샘플링 주파수로부터 생성
            N = length(acc_x);
            t = (0:N-1) / fs;
        end
        
        % 샘플링 주파수 자동 계산 (시간 벡터가 있는 경우)
        if exist('t', 'var') && length(t) > 1
            dt = mean(diff(t));
            if dt > 0
                fs = 1 / dt;
                fprintf('샘플링 주파수 자동 계산: %.2f Hz\n', fs);
            end
        end
        
        N = length(acc_x);
        duration = N / fs;
        
        fprintf('데이터 로드 완료: %d 샘플, %.2f 초\n', N, duration);
        
    catch ME
        fprintf('CSV 파일 로드 오류: %s\n', ME.message);
        fprintf('샘플 데이터를 생성합니다.\n');
        use_sample_data = true;
    end
else
    % 방법 2: MAT 파일에서 로드
    mat_file_path = '';  % 예: 'data/strap_A_data.mat'
    
    if ~isempty(mat_file_path) && exist(mat_file_path, 'file')
        fprintf('MAT 파일에서 데이터 로드 중: %s\n', mat_file_path);
        try
            loaded_data = load(mat_file_path);
            
            % 변수명 매핑 (다양한 형식 지원)
            if isfield(loaded_data, 'acc_x')
                acc_x = loaded_data.acc_x;
                acc_y = loaded_data.acc_y;
                acc_z = loaded_data.acc_z;
            elseif isfield(loaded_data, 'ACC_X')
                acc_x = loaded_data.ACC_X;
                acc_y = loaded_data.ACC_Y;
                acc_z = loaded_data.ACC_Z;
            elseif isfield(loaded_data, 'x')
                acc_x = loaded_data.x;
                acc_y = loaded_data.y;
                acc_z = loaded_data.z;
            else
                % 첫 번째 변수가 X, 다음이 Y, Z라고 가정
                var_names = fieldnames(loaded_data);
                acc_x = loaded_data.(var_names{1});
                acc_y = loaded_data.(var_names{2});
                acc_z = loaded_data.(var_names{3});
            end
            
            % 시간 벡터
            if isfield(loaded_data, 't')
                t = loaded_data.t;
            elseif isfield(loaded_data, 'time')
                t = loaded_data.time;
            else
                N = length(acc_x);
                t = (0:N-1) / fs;
            end
            
            N = length(acc_x);
            duration = N / fs;
            
            fprintf('데이터 로드 완료: %d 샘플, %.2f 초\n', N, duration);
            
        catch ME
            fprintf('MAT 파일 로드 오류: %s\n', ME.message);
            fprintf('샘플 데이터를 생성합니다.\n');
            use_sample_data = true;
        end
    else
        % 방법 3: 샘플 데이터 생성 (데모용)
        fprintf('실제 데이터 파일이 지정되지 않았습니다. 샘플 데이터를 생성합니다.\n');
        fprintf('실제 데이터를 사용하려면 위의 data_file_path 또는 mat_file_path를 설정하세요.\n\n');
        use_sample_data = true;
    end
end

% 샘플 데이터 생성 (데모용)
if exist('use_sample_data', 'var') && use_sample_data
    duration = 60;
    t = 0:1/fs:duration-1/fs;
    N = length(t);
    
    base_freq = 0.5;
    base_amplitude = 0.5;
    noise_level = 0.02;
    high_freq_noise = 0.01;
    
    acc_x = base_amplitude * sin(2*pi*base_freq*t) + ...
            noise_level * randn(size(t)) + ...
            high_freq_noise * sin(2*pi*8*t + rand(size(t))) + ...
            high_freq_noise * 0.5 * sin(2*pi*12*t + rand(size(t)));
    
    acc_y = base_amplitude * cos(2*pi*base_freq*t) + ...
            noise_level * randn(size(t)) + ...
            high_freq_noise * sin(2*pi*8*t + rand(size(t))) + ...
            high_freq_noise * 0.5 * sin(2*pi*12*t + rand(size(t)));
    
    acc_z = base_amplitude * 0.7 * sin(2*pi*base_freq*t + pi/4) + ...
            noise_level * randn(size(t)) + ...
            high_freq_noise * sin(2*pi*8*t + rand(size(t))) + ...
            high_freq_noise * 0.5 * sin(2*pi*12*t + rand(size(t)));
    
    strap_name = 'Sample';
end

fprintf('샘플링 주파수: %.2f Hz\n', fs);
fprintf('측정 시간: %.1f 초\n', duration);
fprintf('총 샘플 수: %d\n\n', N);

%% 3. 통계적 분석
fprintf('=== 통계적 분석 결과 ===\n');

% RMS (Root Mean Square) 값 계산
rms_x = sqrt(mean(acc_x.^2));
rms_y = sqrt(mean(acc_y.^2));
rms_z = sqrt(mean(acc_z.^2));
rms_total = sqrt(mean(acc_x.^2 + acc_y.^2 + acc_z.^2));

% 표준편차 계산
std_x = std(acc_x);
std_y = std(acc_y);
std_z = std(acc_z);
std_total = std(sqrt(acc_x.^2 + acc_y.^2 + acc_z.^2));

% 변동계수 (Coefficient of Variation)
cv_x = std_x / abs(mean(acc_x));
cv_y = std_y / abs(mean(acc_y));
cv_z = std_z / abs(mean(acc_z));

% 평균값
mean_x = mean(acc_x);
mean_y = mean(acc_y);
mean_z = mean(acc_z);

% 최대값, 최소값
max_x = max(acc_x);
max_y = max(acc_y);
max_z = max(acc_z);
min_x = min(acc_x);
min_y = min(acc_y);
min_z = min(acc_z);

fprintf('\n--- RMS 값 ---\n');
fprintf('X=%.4f, Y=%.4f, Z=%.4f, Total=%.4f\n', rms_x, rms_y, rms_z, rms_total);

fprintf('\n--- 표준편차 ---\n');
fprintf('X=%.4f, Y=%.4f, Z=%.4f, Total=%.4f\n', std_x, std_y, std_z, std_total);

fprintf('\n--- 변동계수 ---\n');
fprintf('X=%.4f, Y=%.4f, Z=%.4f\n', cv_x, cv_y, cv_z);

%% 4. 주파수 도메인 분석
fprintf('\n=== 주파수 도메인 분석 ===\n');

% FFT 분석
N_fft = 2^nextpow2(N);
freq = fs * (0:(N_fft/2)) / N_fft;

fft_x = fft(acc_x, N_fft);
fft_y = fft(acc_y, N_fft);
fft_z = fft(acc_z, N_fft);
Pxx_x = abs(fft_x(1:N_fft/2+1)).^2 / N_fft;
Pxx_y = abs(fft_y(1:N_fft/2+1)).^2 / N_fft;
Pxx_z = abs(fft_z(1:N_fft/2+1)).^2 / N_fft;

% 고주파 성분 분석 (미세 진동: 5Hz 이상)
high_freq_idx = freq >= 5;
high_freq_power_x = sum(Pxx_x(high_freq_idx));
high_freq_power_y = sum(Pxx_y(high_freq_idx));
high_freq_power_z = sum(Pxx_z(high_freq_idx));
high_freq_power_total = high_freq_power_x + high_freq_power_y + high_freq_power_z;

fprintf('\n--- 고주파 성분 파워 (5Hz 이상, 미세 진동) ---\n');
fprintf('X=%.6f, Y=%.6f, Z=%.6f, Total=%.6f\n', ...
        high_freq_power_x, high_freq_power_y, high_freq_power_z, high_freq_power_total);

% 주파수 대역별 에너지 분석
freq_bands = [0, 1, 2, 5, 10, 12.5];  % Hz
fprintf('\n--- 주파수 대역별 에너지 분포 ---\n');
band_powers = zeros(length(freq_bands)-1, 1);
for i = 1:length(freq_bands)-1
    band_idx = (freq >= freq_bands(i)) & (freq < freq_bands(i+1));
    band_power = sum(Pxx_x(band_idx) + Pxx_y(band_idx) + Pxx_z(band_idx));
    band_powers(i) = band_power;
    fprintf('%.1f-%.1f Hz: %.6f\n', freq_bands(i), freq_bands(i+1), band_power);
end

%% 5. 시각화
fprintf('\n=== 시각화 생성 중... ===\n');

% Figure 1: 시간 도메인 신호
figure('Position', [100, 100, 1400, 900]);

% X축 신호
subplot(3,3,1);
plot(t(1:min(500, N)), acc_x(1:min(500, N)), 'b-', 'LineWidth', 1.5);
title('ACC X축 신호 (첫 20초)');
xlabel('시간 (s)'); ylabel('가속도 (g)');
grid on;

subplot(3,3,2);
plot(t(1:min(500, N)), acc_y(1:min(500, N)), 'b-', 'LineWidth', 1.5);
title('ACC Y축 신호 (첫 20초)');
xlabel('시간 (s)'); ylabel('가속도 (g)');
grid on;

subplot(3,3,3);
plot(t(1:min(500, N)), acc_z(1:min(500, N)), 'b-', 'LineWidth', 1.5);
title('ACC Z축 신호 (첫 20초)');
xlabel('시간 (s)'); ylabel('가속도 (g)');
grid on;

% 주파수 스펙트럼
subplot(3,3,4);
semilogy(freq, Pxx_x, 'b-', 'LineWidth', 1.5);
title('ACC X축 파워 스펙트럼');
xlabel('주파수 (Hz)'); ylabel('파워');
grid on;
xlim([0 fs/2]);

subplot(3,3,5);
semilogy(freq, Pxx_y, 'b-', 'LineWidth', 1.5);
title('ACC Y축 파워 스펙트럼');
xlabel('주파수 (Hz)'); ylabel('파워');
grid on;
xlim([0 fs/2]);

subplot(3,3,6);
semilogy(freq, Pxx_z, 'b-', 'LineWidth', 1.5);
title('ACC Z축 파워 스펙트럼');
xlabel('주파수 (Hz)'); ylabel('파워');
grid on;
xlim([0 fs/2]);

% 고주파 성분 확대 (미세 진동 분석)
subplot(3,3,7);
high_freq_range = freq >= 5 & freq <= fs/2;
semilogy(freq(high_freq_range), Pxx_x(high_freq_range), 'b-', 'LineWidth', 2);
title('X축 고주파 성분 (5Hz 이상, 미세 진동)');
xlabel('주파수 (Hz)'); ylabel('파워');
grid on;

subplot(3,3,8);
semilogy(freq(high_freq_range), Pxx_y(high_freq_range), 'b-', 'LineWidth', 2);
title('Y축 고주파 성분 (5Hz 이상, 미세 진동)');
xlabel('주파수 (Hz)'); ylabel('파워');
grid on;

subplot(3,3,9);
semilogy(freq(high_freq_range), Pxx_z(high_freq_range), 'b-', 'LineWidth', 2);
title('Z축 고주파 성분 (5Hz 이상, 미세 진동)');
xlabel('주파수 (Hz)'); ylabel('파워');
grid on;

if ~isempty(strap_name)
    sgtitle(sprintf('워치 스트랩 ACC 센서 미세 진동 분석 - %s', strap_name), 'FontSize', 14, 'FontWeight', 'bold');
else
    sgtitle('워치 스트랩 ACC 센서 미세 진동 분석', 'FontSize', 14, 'FontWeight', 'bold');
end

% Figure 2: 통계 및 스펙트로그램
figure('Position', [100, 50, 1400, 900]);

% RMS 및 표준편차
subplot(3,3,1);
bar_data = [rms_x, rms_y, rms_z, rms_total]';
bar(bar_data);
set(gca, 'XTickLabel', {'X', 'Y', 'Z', 'Total'});
ylabel('RMS 값');
title('RMS 값');
grid on;

subplot(3,3,2);
bar_data = [std_x, std_y, std_z, std_total]';
bar(bar_data);
set(gca, 'XTickLabel', {'X', 'Y', 'Z', 'Total'});
ylabel('표준편차');
title('표준편차');
grid on;

% 고주파 파워
subplot(3,3,3);
bar_data = [high_freq_power_x, high_freq_power_y, high_freq_power_z, high_freq_power_total]';
bar(bar_data);
set(gca, 'XTickLabel', {'X', 'Y', 'Z', 'Total'});
ylabel('고주파 파워 (5Hz 이상)');
title('고주파 성분 파워');
grid on;

% 스펙트로그램 - X축
subplot(3,3,4);
[s_x, f_spec, t_spec] = spectrogram(acc_x, round(N/10), round(N/20), [], fs);
imagesc(t_spec, f_spec, 10*log10(abs(s_x).^2));
axis xy; colorbar;
title('X축 스펙트로그램');
xlabel('시간 (s)'); ylabel('주파수 (Hz)');
ylim([0 fs/2]);

% 스펙트로그램 - Y축
subplot(3,3,5);
[s_y, ~, ~] = spectrogram(acc_y, round(N/10), round(N/20), [], fs);
imagesc(t_spec, f_spec, 10*log10(abs(s_y).^2));
axis xy; colorbar;
title('Y축 스펙트로그램');
xlabel('시간 (s)'); ylabel('주파수 (Hz)');
ylim([0 fs/2]);

% 스펙트로그램 - Z축
subplot(3,3,6);
[s_z, ~, ~] = spectrogram(acc_z, round(N/10), round(N/20), [], fs);
imagesc(t_spec, f_spec, 10*log10(abs(s_z).^2));
axis xy; colorbar;
title('Z축 스펙트로그램');
xlabel('시간 (s)'); ylabel('주파수 (Hz)');
ylim([0 fs/2]);

% 주파수 대역별 에너지
subplot(3,3,7);
bar(band_powers);
band_labels_str = cell(length(freq_bands)-1, 1);
for i = 1:length(freq_bands)-1
    band_labels_str{i} = sprintf('%.1f-%.1fHz', freq_bands(i), freq_bands(i+1));
end
set(gca, 'XTickLabel', band_labels_str, 'XTickLabelRotation', 45);
ylabel('에너지');
title('주파수 대역별 에너지 분포');
grid on;

% 히스토그램
subplot(3,3,8);
histogram(acc_x, 50, 'FaceColor', 'r', 'FaceAlpha', 0.5); hold on;
histogram(acc_y, 50, 'FaceColor', 'g', 'FaceAlpha', 0.5);
histogram(acc_z, 50, 'FaceColor', 'b', 'FaceAlpha', 0.5);
xlabel('가속도 (g)'); ylabel('빈도');
title('가속도 분포');
legend('X', 'Y', 'Z', 'Location', 'best');
grid on;

% 박스플롯
subplot(3,3,9);
boxplot([acc_x', acc_y', acc_z'], 'Labels', {'X', 'Y', 'Z'});
ylabel('가속도 (g)');
title('가속도 박스플롯');
grid on;

if ~isempty(strap_name)
    sgtitle(sprintf('통계 및 시간-주파수 분석 - %s', strap_name), 'FontSize', 14, 'FontWeight', 'bold');
else
    sgtitle('통계 및 시간-주파수 분석', 'FontSize', 14, 'FontWeight', 'bold');
end

% Figure 3: 총 가속도 벡터 크기 분석
figure('Position', [100, 0, 1400, 600]);

% 총 가속도 벡터 크기
acc_magnitude = sqrt(acc_x.^2 + acc_y.^2 + acc_z.^2);

subplot(2,3,1);
plot(t(1:min(500, N)), acc_magnitude(1:min(500, N)), 'b-', 'LineWidth', 1.5);
title('총 가속도 벡터 크기');
xlabel('시간 (s)'); ylabel('가속도 크기 (g)');
grid on;

subplot(2,3,2);
histogram(acc_magnitude, 50, 'FaceColor', 'b', 'FaceAlpha', 0.7);
title('가속도 크기 분포');
xlabel('가속도 크기 (g)'); ylabel('빈도');
grid on;

subplot(2,3,3);
boxplot(acc_magnitude', 'Labels', {'Magnitude'});
title('가속도 크기 박스플롯');
ylabel('가속도 크기 (g)');
grid on;

% 고주파 성분 필터링 (고역 통과 필터)
% 5Hz 고역 통과 필터 설계
[b, a] = butter(4, 5/(fs/2), 'high');
acc_x_high = filter(b, a, acc_x);
acc_y_high = filter(b, a, acc_y);
acc_z_high = filter(b, a, acc_z);

acc_magnitude_high = sqrt(acc_x_high.^2 + acc_y_high.^2 + acc_z_high.^2);

subplot(2,3,4);
plot(t(1:min(500, N)), acc_magnitude_high(1:min(500, N)), 'b-', 'LineWidth', 1.5);
title('고주파 성분만 추출 (5Hz 이상)');
xlabel('시간 (s)'); ylabel('가속도 크기 (g)');
grid on;

subplot(2,3,5);
histogram(acc_magnitude_high, 50, 'FaceColor', 'b', 'FaceAlpha', 0.7);
title('고주파 성분 분포');
xlabel('가속도 크기 (g)'); ylabel('빈도');
grid on;

subplot(2,3,6);
boxplot(acc_magnitude_high', 'Labels', {'High Freq'});
title('고주파 성분 박스플롯');
ylabel('가속도 크기 (g)');
grid on;

if ~isempty(strap_name)
    sgtitle(sprintf('총 가속도 벡터 및 고주파 성분 분석 - %s', strap_name), 'FontSize', 14, 'FontWeight', 'bold');
else
    sgtitle('총 가속도 벡터 및 고주파 성분 분석', 'FontSize', 14, 'FontWeight', 'bold');
end

%% 6. 결과 요약 및 저장
fprintf('\n=== 분석 결과 요약 ===\n');
fprintf('RMS 값: %.4f\n', rms_total);
fprintf('표준편차: %.4f\n', std_total);
fprintf('고주파 파워 (5Hz 이상): %.6f\n', high_freq_power_total);

% 결과 구조체 생성
results.rms = struct('x', rms_x, 'y', rms_y, 'z', rms_z, 'total', rms_total);
results.std = struct('x', std_x, 'y', std_y, 'z', std_z, 'total', std_total);
results.mean = struct('x', mean_x, 'y', mean_y, 'z', mean_z);
results.max = struct('x', max_x, 'y', max_y, 'z', max_z);
results.min = struct('x', min_x, 'y', min_y, 'z', min_z);
results.cv = struct('x', cv_x, 'y', cv_y, 'z', cv_z);
results.high_freq_power = struct('x', high_freq_power_x, 'y', high_freq_power_y, ...
                                  'z', high_freq_power_z, 'total', high_freq_power_total);
results.fs = fs;
results.duration = duration;
results.freq = freq;
results.Pxx = struct('x', Pxx_x, 'y', Pxx_y, 'z', Pxx_z);
results.freq_bands = freq_bands;
results.band_powers = band_powers;

% MAT 파일 저장
if ~isempty(strap_name)
    mat_filename = sprintf('strap_acc_vibration_results_%s.mat', strap_name);
else
    mat_filename = 'strap_acc_vibration_results.mat';
end
save(mat_filename, 'results', 'acc_x', 'acc_y', 'acc_z', 't', 'freq');
fprintf('\n결과가 ''%s'' 파일에 저장되었습니다.\n', mat_filename);

% CSV 파일로 결과 저장 (비교를 위해)
if ~isempty(strap_name)
    csv_filename = sprintf('strap_acc_vibration_results_%s.csv', strap_name);
else
    csv_filename = 'strap_acc_vibration_results.csv';
end

% 통계 결과를 CSV로 저장
statistics_table = table(...
    {'RMS_X'; 'RMS_Y'; 'RMS_Z'; 'RMS_Total'; ...
     'Std_X'; 'Std_Y'; 'Std_Z'; 'Std_Total'; ...
     'Mean_X'; 'Mean_Y'; 'Mean_Z'; ...
     'Max_X'; 'Max_Y'; 'Max_Z'; ...
     'Min_X'; 'Min_Y'; 'Min_Z'; ...
     'CV_X'; 'CV_Y'; 'CV_Z'; ...
     'HighFreqPower_X'; 'HighFreqPower_Y'; 'HighFreqPower_Z'; 'HighFreqPower_Total'}, ...
    [rms_x; rms_y; rms_z; rms_total; ...
     std_x; std_y; std_z; std_total; ...
     mean_x; mean_y; mean_z; ...
     max_x; max_y; max_z; ...
     min_x; min_y; min_z; ...
     cv_x; cv_y; cv_z; ...
     high_freq_power_x; high_freq_power_y; high_freq_power_z; high_freq_power_total], ...
    'VariableNames', {'Metric', 'Value'});

writetable(statistics_table, csv_filename);
fprintf('통계 결과가 ''%s'' 파일에 저장되었습니다.\n', csv_filename);

% 주파수 대역별 에너지를 CSV로 저장
band_energy_table = table(...
    cellstr(band_labels_str), ...
    band_powers, ...
    'VariableNames', {'FrequencyBand', 'Energy'});

if ~isempty(strap_name)
    band_csv_filename = sprintf('strap_freq_band_energy_%s.csv', strap_name);
else
    band_csv_filename = 'strap_freq_band_energy.csv';
end
writetable(band_energy_table, band_csv_filename);
fprintf('주파수 대역별 에너지가 ''%s'' 파일에 저장되었습니다.\n', band_csv_filename);

% 원본 데이터도 CSV로 저장 (선택사항)
if ~isempty(strap_name)
    data_csv_filename = sprintf('strap_acc_data_%s.csv', strap_name);
else
    data_csv_filename = 'strap_acc_data.csv';
end
data_table = table(t', acc_x, acc_y, acc_z, ...
    'VariableNames', {'Time', 'ACC_X', 'ACC_Y', 'ACC_Z'});
writetable(data_table, data_csv_filename);
fprintf('원본 데이터가 ''%s'' 파일에 저장되었습니다.\n', data_csv_filename);

fprintf('\n분석 완료!\n');
fprintf('비교를 위해 각 스트랩의 CSV 파일을 사용하세요.\n');
