%% 신호 분석 비교: STFT vs 필터뱅크
% 0.2Hz 단위로 0.2~4Hz 대역을 분석
clear all; close all; clc;

%% 1. 신호 생성
fs = 100;  % 샘플링 주파수 (Hz)
duration = 60;  % 신호 길이 (초)
t = 0:1/fs:duration-1/fs;

% 테스트 신호 생성 (여러 주파수 성분 포함)
signal = 0.5*sin(2*pi*0.5*t) + ...    % 0.5 Hz
         0.3*sin(2*pi*1.2*t) + ...    % 1.2 Hz
         0.4*sin(2*pi*2.5*t) + ...    % 2.5 Hz
         0.2*sin(2*pi*3.8*t) + ...    % 3.8 Hz
         0.1*randn(size(t));          % 잡음

%% 2. 주파수 대역 설정
freq_start = 0.2;  % 시작 주파수 (Hz)
freq_end = 4.0;    % 끝 주파수 (Hz)
freq_step = 0.2;   % 주파수 간격 (Hz)
freq_bands = freq_start:freq_step:freq_end;  % 19개 주파수 대역
n_bands = length(freq_bands);

fprintf('분석 주파수 대역: %.1f Hz ~ %.1f Hz (%.1f Hz 간격, %d개 대역)\n', ...
    freq_start, freq_end, freq_step, n_bands);

%% 3. STFT 분석
% STFT 파라미터 설정
window_length = 10 * fs;  % 10초 윈도우 (0.1Hz 해상도)
overlap = window_length * 0.9;  % 90% 오버랩
nfft = 2^nextpow2(window_length * 4);  % FFT 크기

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

% 각 주파수 대역에 대한 밴드패스 필터 설계 및 적용
for i = 1:n_bands
    % 밴드패스 필터 설계 (Butterworth 4차)
    if i == 1
        % 첫 번째 대역: 0.1 ~ 0.3 Hz
        [b, a] = butter(4, [0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    elseif i == n_bands
        % 마지막 대역: 3.9 ~ 4.1 Hz
        [b, a] = butter(4, [freq_bands(i)-0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
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
    filterbank_power_downsampled(i,:) = decimate(filterbank_power(i,:), downsample_factor);
end

%% 5. 결과 시각화
figure('Position', [100, 100, 1400, 900]);

% 5.1 원본 신호
subplot(3,2,1);
plot(t, signal);
xlabel('시간 (초)');
ylabel('진폭');
title('원본 신호');
grid on;
xlim([0, 60]);

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
imagesc(T, freq_bands, 10*log10(stft_power));
axis xy;
xlabel('시간 (초)');
ylabel('주파수 (Hz)');
title('STFT 분석 결과');
colorbar;
caxis([-40, 0]);  % dB 스케일 조정

% 5.4 필터뱅크 결과
subplot(3,2,4);
imagesc(T, freq_bands, 10*log10(filterbank_power_downsampled));
axis xy;
xlabel('시간 (초)');
ylabel('주파수 (Hz)');
title('필터뱅크 분석 결과');
colorbar;
caxis([-40, 0]);  % dB 스케일 조정

% 5.5 특정 주파수 대역 비교 (1.2 Hz)
target_freq_idx = find(freq_bands == 1.2);
subplot(3,2,5);
plot(T, 10*log10(stft_power(target_freq_idx,:)), 'b-', 'LineWidth', 2);
hold on;
plot(T, 10*log10(filterbank_power_downsampled(target_freq_idx,:)), 'r--', 'LineWidth', 2);
xlabel('시간 (초)');
ylabel('파워 (dB)');
title('1.2 Hz 대역 파워 비교');
legend('STFT', '필터뱅크', 'Location', 'best');
grid on;

% 5.6 전체 대역 평균 파워 비교
subplot(3,2,6);
mean_stft = mean(stft_power, 2);
mean_filterbank = mean(filterbank_power_downsampled, 2);
plot(freq_bands, 10*log10(mean_stft), 'b-o', 'LineWidth', 2);
hold on;
plot(freq_bands, 10*log10(mean_filterbank), 'r--s', 'LineWidth', 2);
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
    correlations(i) = corr(stft_power(i,:)', filterbank_power_downsampled(i,:)');
end
fprintf('평균 상관계수: %.3f\n', mean(correlations));

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
        [b, a] = butter(4, [freq_bands(i)-0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    else
        [b, a] = butter(4, [freq_bands(i)-0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    end
    filtered = filtfilt(b, a, signal);
    analytic = hilbert(filtered);
end
fb_time = toc;

fprintf('\nSTFT 계산 시간: %.3f 초\n', stft_time);
fprintf('필터뱅크 계산 시간: %.3f 초\n', fb_time);

%% 7. 추가 분석: 시간-주파수 해상도 비교
figure('Position', [100, 100, 1200, 600]);

% 7.1 짧은 시간 구간 확대
time_range = [20, 30];  % 20~30초 구간
time_idx = find(T >= time_range(1) & T <= time_range(2));

subplot(2,1,1);
imagesc(T(time_idx), freq_bands, 10*log10(stft_power(:,time_idx)));
axis xy;
xlabel('시간 (초)');
ylabel('주파수 (Hz)');
title('STFT - 시간 구간 확대 (20-30초)');
colorbar;
caxis([-40, 0]);

subplot(2,1,2);
imagesc(T(time_idx), freq_bands, 10*log10(filterbank_power_downsampled(:,time_idx)));
axis xy;
xlabel('시간 (초)');
ylabel('주파수 (Hz)');
title('필터뱅크 - 시간 구간 확대 (20-30초)');
colorbar;
caxis([-40, 0]);

%% 8. 분석 결과 요약
fprintf('\n=== 분석 결과 요약 ===\n');
fprintf('1. STFT 방식:\n');
fprintf('   - 장점: 균일한 시간-주파수 해상도, 빠른 계산 (FFT 활용)\n');
fprintf('   - 단점: 고정된 윈도우 크기로 인한 해상도 제한\n');
fprintf('\n2. 필터뱅크 방식:\n');
fprintf('   - 장점: 각 대역별 독립적 처리, 실시간 처리 가능\n');
fprintf('   - 단점: 필터 설계의 복잡성, 대역 간 중첩 문제\n');
fprintf('\n3. 두 방법의 상관성: %.1f%%\n', mean(correlations)*100);