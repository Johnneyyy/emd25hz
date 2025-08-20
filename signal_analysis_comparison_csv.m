%% 신호 분석 비교: STFT vs 필터뱅크 (CSV 입력)
% CSV 파일에서 신호를 읽어 0.2Hz 단위로 0.2~4Hz 대역을 분석
clear all; close all; clc;

%% 1. CSV 파일에서 신호 읽기
% CSV 파일 선택
[filename, pathname] = uigetfile('*.csv', 'CSV 신호 파일 선택');
if isequal(filename, 0)
    error('CSV 파일을 선택해주세요.');
end

filepath = fullfile(pathname, filename);

% CSV 파일 읽기 시도
try
    % 헤더가 있는 경우
    data = readtable(filepath);
    if width(data) >= 2
        t = data{:,1};
        signal = data{:,2};
    else
        error('CSV 파일은 최소 2개의 열(시간, 신호)을 포함해야 합니다.');
    end
catch
    % 헤더가 없는 경우
    try
        data = csvread(filepath);
        t = data(:,1);
        signal = data(:,2);
    catch
        error('CSV 파일을 읽을 수 없습니다. 형식을 확인하세요.');
    end
end

% 전치 행렬로 변환 (행 벡터로 만들기)
t = t(:)';
signal = signal(:)';

% 샘플링 주파수 계산
dt = mean(diff(t));
fs = round(1/dt);
duration = t(end) - t(1);

fprintf('\n=== 신호 정보 ===\n');
fprintf('파일명: %s\n', filename);
fprintf('신호 길이: %.2f 초\n', duration);
fprintf('샘플 수: %d\n', length(signal));
fprintf('샘플링 주파수: %d Hz\n', fs);

%% 2. 주파수 대역 설정
freq_start = 0.2;  % 시작 주파수 (Hz)
freq_end = 4.0;    % 끝 주파수 (Hz)
freq_step = 0.2;   % 주파수 간격 (Hz)
freq_bands = freq_start:freq_step:freq_end;  % 19개 주파수 대역
n_bands = length(freq_bands);

fprintf('\n분석 주파수 대역: %.1f Hz ~ %.1f Hz (%.1f Hz 간격, %d개 대역)\n', ...
    freq_start, freq_end, freq_step, n_bands);

%% 3. STFT 분석
% STFT 파라미터 설정 (신호 길이에 따라 적응적으로 설정)
if duration < 30
    window_duration = 5;  % 짧은 신호: 5초 윈도우
elseif duration < 120
    window_duration = 10;  % 중간 신호: 10초 윈도우
else
    window_duration = 20;  % 긴 신호: 20초 윈도우
end

window_length = min(window_duration * fs, length(signal)/2);
overlap = round(window_length * 0.9);  % 90% 오버랩
nfft = 2^nextpow2(window_length * 4);  % FFT 크기

fprintf('\nSTFT 파라미터: 윈도우 크기 = %.1f초 (%d 샘플)\n', window_length/fs, window_length);

% STFT 수행
[S, F, T] = spectrogram(signal, hamming(window_length), overlap, nfft, fs);

% 각 주파수 대역의 파워 추출
stft_power = zeros(n_bands, length(T));
for i = 1:n_bands
    % 각 대역의 중심 주파수 주변 ±0.1Hz 범위의 파워 평균
    freq_idx = find(F >= (freq_bands(i)-0.1) & F <= (freq_bands(i)+0.1));
    if ~isempty(freq_idx)
        stft_power(i,:) = mean(abs(S(freq_idx,:)).^2, 1);
    end
end

%% 4. 필터뱅크 분석
% 필터뱅크 파워 저장
filterbank_power = zeros(n_bands, length(t));

fprintf('필터뱅크 분석 진행 중...\n');
% 각 주파수 대역에 대한 밴드패스 필터 설계 및 적용
for i = 1:n_bands
    % 진행 상황 표시
    if mod(i, 5) == 0
        fprintf('  %d/%d 대역 처리 중...\n', i, n_bands);
    end
    
    % 밴드패스 필터 설계 (Butterworth 4차)
    if i == 1
        % 첫 번째 대역: 0.1 ~ 0.3 Hz
        [b, a] = butter(4, [0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    elseif i == n_bands
        % 마지막 대역: 3.9 ~ 4.1 Hz
        [b, a] = butter(4, [freq_bands(i)-0.1, min(freq_bands(i)+0.1, fs/2-0.1)]/(fs/2), 'bandpass');
    else
        % 중간 대역: 중심주파수 ±0.1 Hz
        [b, a] = butter(4, [freq_bands(i)-0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    end
    
    % 필터 적용
    filtered_signal = filtfilt(b, a, signal);
    
    % 순시 파워 계산 (Hilbert 변환 사용)
    analytic_signal = hilbert(filtered_signal);
    filterbank_power(i,:) = abs(analytic_signal).^2;
end

% 필터뱅크 결과를 STFT와 동일한 시간 해상도로 다운샘플링
downsample_factor = round(length(t) / length(T));
filterbank_power_downsampled = zeros(n_bands, length(T));
for i = 1:n_bands
    if downsample_factor > 1
        filterbank_power_downsampled(i,:) = decimate(filterbank_power(i,:), downsample_factor);
    else
        % 보간이 필요한 경우
        filterbank_power_downsampled(i,:) = interp1(1:length(filterbank_power(i,:)), ...
            filterbank_power(i,:), linspace(1, length(filterbank_power(i,:)), length(T)));
    end
end

%% 5. 결과 시각화
figure('Position', [100, 100, 1400, 900]);
sgtitle(sprintf('신호 분석 비교: %s', filename), 'Interpreter', 'none');

% 5.1 원본 신호
subplot(3,2,1);
plot(t, signal);
xlabel('시간 (초)');
ylabel('진폭');
title('원본 신호');
grid on;
xlim([t(1), t(end)]);

% 5.2 신호의 FFT 스펙트럼
subplot(3,2,2);
Y = fft(signal);
f_fft = fs*(0:(length(signal)/2))/length(signal);
P = abs(Y/length(signal));
P = P(1:length(signal)/2+1);
P(2:end-1) = 2*P(2:end-1);
plot(f_fft, P);
xlabel('주파수 (Hz)');
ylabel('진폭');
title('신호의 주파수 스펙트럼');
grid on;
xlim([0, 5]);

% 5.3 STFT 결과 (스펙트로그램)
subplot(3,2,3);
imagesc(T, freq_bands, 10*log10(stft_power + eps));
axis xy;
xlabel('시간 (초)');
ylabel('주파수 (Hz)');
title('STFT 분석 결과');
colorbar;
caxis([-60, max(10*log10(stft_power(:)))]);

% 5.4 필터뱅크 결과
subplot(3,2,4);
imagesc(T, freq_bands, 10*log10(filterbank_power_downsampled + eps));
axis xy;
xlabel('시간 (초)');
ylabel('주파수 (Hz)');
title('필터뱅크 분석 결과');
colorbar;
caxis([-60, max(10*log10(filterbank_power_downsampled(:)))]);

% 5.5 특정 주파수 대역 비교
% 가장 강한 파워를 가진 주파수 대역 찾기
[~, max_power_idx] = max(mean(stft_power, 2));
target_freq = freq_bands(max_power_idx);

subplot(3,2,5);
plot(T, 10*log10(stft_power(max_power_idx,:) + eps), 'b-', 'LineWidth', 2);
hold on;
plot(T, 10*log10(filterbank_power_downsampled(max_power_idx,:) + eps), 'r--', 'LineWidth', 2);
xlabel('시간 (초)');
ylabel('파워 (dB)');
title(sprintf('%.1f Hz 대역 파워 비교', target_freq));
legend('STFT', '필터뱅크', 'Location', 'best');
grid on;

% 5.6 전체 대역 평균 파워 비교
subplot(3,2,6);
mean_stft = mean(stft_power, 2);
mean_filterbank = mean(filterbank_power_downsampled, 2);
plot(freq_bands, 10*log10(mean_stft + eps), 'b-o', 'LineWidth', 2);
hold on;
plot(freq_bands, 10*log10(mean_filterbank + eps), 'r--s', 'LineWidth', 2);
xlabel('주파수 (Hz)');
ylabel('평균 파워 (dB)');
title('주파수별 평균 파워 비교');
legend('STFT', '필터뱅크', 'Location', 'best');
grid on;

%% 6. 정량적 비교
fprintf('\n=== 분석 방법 비교 ===\n');

% 6.1 상관계수 계산
correlations = zeros(n_bands, 1);
for i = 1:n_bands
    if ~any(isnan(stft_power(i,:))) && ~any(isnan(filterbank_power_downsampled(i,:)))
        correlations(i) = corr(stft_power(i,:)', filterbank_power_downsampled(i,:)');
    end
end
valid_corr = correlations(~isnan(correlations));
fprintf('평균 상관계수: %.3f\n', mean(valid_corr));

% 6.2 주요 주파수 성분 검출 비교
[~, stft_peak_idx] = max(mean_stft);
[~, fb_peak_idx] = max(mean_filterbank);
fprintf('STFT 최대 파워 주파수: %.1f Hz\n', freq_bands(stft_peak_idx));
fprintf('필터뱅크 최대 파워 주파수: %.1f Hz\n', freq_bands(fb_peak_idx));

% 6.3 계산 시간 비교
tic;
spectrogram(signal, hamming(window_length), overlap, nfft, fs);
stft_time = toc;

tic;
for i = 1:n_bands
    if i == 1
        [b, a] = butter(4, [0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    elseif i == n_bands
        [b, a] = butter(4, [freq_bands(i)-0.1, min(freq_bands(i)+0.1, fs/2-0.1)]/(fs/2), 'bandpass');
    else
        [b, a] = butter(4, [freq_bands(i)-0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    end
    filtered = filtfilt(b, a, signal);
    analytic = hilbert(filtered);
end
fb_time = toc;

fprintf('\nSTFT 계산 시간: %.3f 초\n', stft_time);
fprintf('필터뱅크 계산 시간: %.3f 초\n', fb_time);

%% 7. 결과 저장
% 분석 결과를 CSV 파일로 저장
save_results = questdlg('분석 결과를 CSV 파일로 저장하시겠습니까?', ...
    '결과 저장', '예', '아니오', '예');

if strcmp(save_results, '예')
    % STFT 결과 저장
    stft_filename = strrep(filename, '.csv', '_stft_results.csv');
    stft_data = [T; stft_power];
    stft_header = ['Time(s)'];
    for i = 1:n_bands
        stft_header = [stft_header, sprintf(',%.1fHz', freq_bands(i))];
    end
    
    fid = fopen(stft_filename, 'w');
    fprintf(fid, '%s\n', stft_header);
    fclose(fid);
    dlmwrite(stft_filename, stft_data', '-append', 'delimiter', ',', 'precision', 6);
    
    % 필터뱅크 결과 저장
    fb_filename = strrep(filename, '.csv', '_filterbank_results.csv');
    fb_data = [T; filterbank_power_downsampled];
    
    fid = fopen(fb_filename, 'w');
    fprintf(fid, '%s\n', stft_header);  % 같은 헤더 사용
    fclose(fid);
    dlmwrite(fb_filename, fb_data', '-append', 'delimiter', ',', 'precision', 6);
    
    fprintf('\n결과가 저장되었습니다:\n');
    fprintf('STFT 결과: %s\n', stft_filename);
    fprintf('필터뱅크 결과: %s\n', fb_filename);
end

%% 8. 추가 분석: 시간-주파수 해상도 비교
figure('Position', [100, 100, 1200, 600]);
sgtitle('시간-주파수 해상도 비교', 'FontSize', 14);

% 8.1 짧은 시간 구간 확대
time_range_percent = 0.3;  % 전체 시간의 30% 구간
time_center = duration / 2;
time_window = duration * time_range_percent;
time_range = [time_center - time_window/2, time_center + time_window/2];
time_idx = find(T >= time_range(1) & T <= time_range(2));

if ~isempty(time_idx)
    subplot(2,1,1);
    imagesc(T(time_idx), freq_bands, 10*log10(stft_power(:,time_idx) + eps));
    axis xy;
    xlabel('시간 (초)');
    ylabel('주파수 (Hz)');
    title(sprintf('STFT - 시간 구간 확대 (%.1f-%.1f초)', time_range(1), time_range(2)));
    colorbar;
    caxis([-60, max(10*log10(stft_power(:)))]);
    
    subplot(2,1,2);
    imagesc(T(time_idx), freq_bands, 10*log10(filterbank_power_downsampled(:,time_idx) + eps));
    axis xy;
    xlabel('시간 (초)');
    ylabel('주파수 (Hz)');
    title(sprintf('필터뱅크 - 시간 구간 확대 (%.1f-%.1f초)', time_range(1), time_range(2)));
    colorbar;
    caxis([-60, max(10*log10(filterbank_power_downsampled(:)))]);
end

%% 9. 분석 결과 요약
fprintf('\n=== 분석 결과 요약 ===\n');
fprintf('1. STFT 방식:\n');
fprintf('   - 장점: 균일한 시간-주파수 해상도, 빠른 계산 (FFT 활용)\n');
fprintf('   - 단점: 고정된 윈도우 크기로 인한 해상도 제한\n');
fprintf('\n2. 필터뱅크 방식:\n');
fprintf('   - 장점: 각 대역별 독립적 처리, 실시간 처리 가능\n');
fprintf('   - 단점: 필터 설계의 복잡성, 대역 간 중첩 문제\n');
fprintf('\n3. 두 방법의 상관성: %.1f%%\n', mean(valid_corr)*100);