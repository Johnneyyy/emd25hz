%% 고급 신호 분석: STFT vs 필터뱅크 상세 비교
% 다양한 신호 유형에 대한 분석 성능 비교
clear all; close all; clc;

%% 1. 다양한 테스트 신호 생성
fs = 100;  % 샘플링 주파수 (Hz)
duration = 60;  % 신호 길이 (초)
t = 0:1/fs:duration-1/fs;

% 1.1 정상 신호 (Stationary signal)
signal_stationary = 0.5*sin(2*pi*0.5*t) + 0.3*sin(2*pi*1.2*t) + ...
                   0.4*sin(2*pi*2.5*t) + 0.2*sin(2*pi*3.8*t);

% 1.2 주파수 변조 신호 (Chirp signal)
f0 = 0.5; f1 = 3.5;
signal_chirp = chirp(t, f0, duration, f1);

% 1.3 시변 신호 (Time-varying signal)
signal_timevar = zeros(size(t));
for i = 1:length(t)
    if t(i) < 20
        signal_timevar(i) = sin(2*pi*0.8*t(i));
    elseif t(i) < 40
        signal_timevar(i) = sin(2*pi*2.0*t(i));
    else
        signal_timevar(i) = sin(2*pi*3.5*t(i));
    end
end

% 1.4 펄스 신호
signal_pulse = zeros(size(t));
pulse_times = [10, 25, 40, 55];
for pt = pulse_times
    pulse_idx = find(abs(t - pt) < 0.5);
    signal_pulse(pulse_idx) = 1;
end

% 잡음 추가
noise_level = 0.1;
signals = {signal_stationary + noise_level*randn(size(t)), ...
          signal_chirp + noise_level*randn(size(t)), ...
          signal_timevar + noise_level*randn(size(t)), ...
          signal_pulse + noise_level*randn(size(t))};
signal_names = {'정상 신호', '주파수 변조 신호', '시변 신호', '펄스 신호'};

%% 2. 분석 파라미터 설정
freq_start = 0.2;
freq_end = 4.0;
freq_step = 0.2;
freq_bands = freq_start:freq_step:freq_end;
n_bands = length(freq_bands);

% STFT 파라미터
window_lengths = [5*fs, 10*fs, 20*fs];  % 다양한 윈도우 크기
window_names = {'5초', '10초', '20초'};

%% 3. 각 신호에 대한 분석 수행
for sig_idx = 1:length(signals)
    signal = signals{sig_idx};
    
    fprintf('\n=== %s 분석 중... ===\n', signal_names{sig_idx});
    
    figure('Position', [100, 100, 1600, 900]);
    sgtitle(sprintf('%s 분석 결과', signal_names{sig_idx}));
    
    % 3.1 원본 신호 표시
    subplot(3,3,1);
    plot(t, signal);
    xlabel('시간 (초)');
    ylabel('진폭');
    title('원본 신호');
    grid on;
    xlim([0, 60]);
    
    % 3.2 다양한 윈도우 크기의 STFT
    for win_idx = 1:length(window_lengths)
        window_length = window_lengths(win_idx);
        overlap = window_length * 0.9;
        nfft = 2^nextpow2(window_length * 4);
        
        [S, F, T] = spectrogram(signal, hamming(window_length), overlap, nfft, fs);
        
        % 주파수 대역별 파워 추출
        stft_power = zeros(n_bands, length(T));
        for i = 1:n_bands
            freq_idx = find(F >= (freq_bands(i)-0.1) & F <= (freq_bands(i)+0.1));
            if ~isempty(freq_idx)
                stft_power(i,:) = mean(abs(S(freq_idx,:)).^2, 1);
            end
        end
        
        subplot(3,3,1+win_idx);
        imagesc(T, freq_bands, 10*log10(stft_power + eps));
        axis xy;
        xlabel('시간 (초)');
        ylabel('주파수 (Hz)');
        title(sprintf('STFT (%s 윈도우)', window_names{win_idx}));
        colorbar;
        caxis([-60, 0]);
    end
    
    % 3.3 다양한 필터 차수의 필터뱅크
    filter_orders = [2, 4, 6];
    
    for ord_idx = 1:length(filter_orders)
        filter_order = filter_orders(ord_idx);
        filterbank_power = zeros(n_bands, length(t));
        
        for i = 1:n_bands
            if i == 1
                [b, a] = butter(filter_order, [0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
            elseif i == n_bands
                [b, a] = butter(filter_order, [freq_bands(i)-0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
            else
                [b, a] = butter(filter_order, [freq_bands(i)-0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
            end
            
            filtered_signal = filtfilt(b, a, signal);
            analytic_signal = hilbert(filtered_signal);
            filterbank_power(i,:) = abs(analytic_signal).^2;
        end
        
        % 다운샘플링 (표시용)
        downsample_factor = 100;
        t_down = t(1:downsample_factor:end);
        fb_power_down = filterbank_power(:, 1:downsample_factor:end);
        
        subplot(3,3,4+ord_idx);
        imagesc(t_down, freq_bands, 10*log10(fb_power_down + eps));
        axis xy;
        xlabel('시간 (초)');
        ylabel('주파수 (Hz)');
        title(sprintf('필터뱅크 (%d차)', filter_order));
        colorbar;
        caxis([-60, 0]);
    end
    
    % 3.4 성능 메트릭 계산 및 표시
    subplot(3,3,8);
    % 주파수 해상도 비교
    text(0.1, 0.9, '성능 메트릭:', 'FontWeight', 'bold', 'FontSize', 12);
    text(0.1, 0.7, sprintf('신호 유형: %s', signal_names{sig_idx}));
    text(0.1, 0.5, '주파수 해상도: STFT는 윈도우 크기에 반비례');
    text(0.1, 0.3, '시간 해상도: 필터뱅크가 더 우수');
    text(0.1, 0.1, '계산 복잡도: STFT < 필터뱅크');
    axis off;
    
    % 3.5 스펙트럼 비교
    subplot(3,3,9);
    Y = fft(signal);
    f_fft = fs*(0:(length(signal)/2))/length(signal);
    P = abs(Y/length(signal));
    P = P(1:length(signal)/2+1);
    P(2:end-1) = 2*P(2:end-1);
    plot(f_fft, 20*log10(P + eps));
    xlabel('주파수 (Hz)');
    ylabel('파워 (dB)');
    title('전체 스펙트럼');
    grid on;
    xlim([0, 5]);
    ylim([-60, 0]);
end

%% 4. 정량적 성능 비교
figure('Position', [100, 100, 1200, 800]);

% 4.1 시간-주파수 불확정성 원리 시각화
subplot(2,2,1);
window_sizes = [2, 5, 10, 20, 40];  % 초 단위
freq_resolution = fs ./ (window_sizes * fs);  % Hz
time_resolution = window_sizes;  % 초

plot(time_resolution, freq_resolution, 'bo-', 'LineWidth', 2);
xlabel('시간 해상도 (초)');
ylabel('주파수 해상도 (Hz)');
title('시간-주파수 해상도 트레이드오프');
grid on;
text(10, 0.1, '불확정성 원리: Δt × Δf ≥ 1/(4π)', 'FontSize', 10);

% 4.2 필터뱅크 주파수 응답
subplot(2,2,2);
freqz_points = 1024;
H_total = zeros(n_bands, freqz_points);

for i = 1:n_bands
    if i == 1
        [b, a] = butter(4, [0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    elseif i == n_bands
        [b, a] = butter(4, [freq_bands(i)-0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    else
        [b, a] = butter(4, [freq_bands(i)-0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    end
    [H, W] = freqz(b, a, freqz_points);
    H_total(i,:) = abs(H);
end

f_freqz = W * fs / (2*pi);
plot(f_freqz, 20*log10(H_total'));
xlabel('주파수 (Hz)');
ylabel('크기 응답 (dB)');
title('필터뱅크 주파수 응답');
grid on;
xlim([0, 5]);
ylim([-60, 5]);

% 4.3 계산 시간 비교
subplot(2,2,3);
signal_lengths = [10, 30, 60, 120, 300];  % 초 단위
stft_times = zeros(size(signal_lengths));
fb_times = zeros(size(signal_lengths));

for i = 1:length(signal_lengths)
    test_signal = randn(1, signal_lengths(i)*fs);
    
    % STFT 시간
    tic;
    spectrogram(test_signal, hamming(10*fs), 9*fs, 2048, fs);
    stft_times(i) = toc;
    
    % 필터뱅크 시간
    tic;
    for j = 1:n_bands
        [b, a] = butter(4, [freq_bands(j)-0.1, freq_bands(j)+0.1]/(fs/2), 'bandpass');
        filtered = filtfilt(b, a, test_signal);
        hilbert(filtered);
    end
    fb_times(i) = toc;
end

plot(signal_lengths, stft_times, 'b-o', 'LineWidth', 2);
hold on;
plot(signal_lengths, fb_times, 'r--s', 'LineWidth', 2);
xlabel('신호 길이 (초)');
ylabel('계산 시간 (초)');
title('계산 시간 비교');
legend('STFT', '필터뱅크', 'Location', 'northwest');
grid on;

% 4.4 메모리 사용량 추정
subplot(2,2,4);
% STFT 메모리: 스펙트로그램 행렬
stft_memory = signal_lengths * fs * 8 * 1024 / (1024^2);  % MB (추정)
% 필터뱅크 메모리: 필터 계수 + 필터링된 신호
fb_memory = signal_lengths * fs * 8 * n_bands / (1024^2);  % MB

bar([stft_memory; fb_memory]');
set(gca, 'XTickLabel', signal_lengths);
xlabel('신호 길이 (초)');
ylabel('메모리 사용량 (MB)');
title('메모리 사용량 비교 (추정)');
legend('STFT', '필터뱅크', 'Location', 'northwest');
grid on;

%% 5. 최종 요약
fprintf('\n=== 최종 분석 요약 ===\n');
fprintf('1. STFT 방식:\n');
fprintf('   - 최적 사용 조건: 정상 신호, 주파수 내용이 천천히 변하는 신호\n');
fprintf('   - 윈도우 크기 선택이 중요 (시간-주파수 해상도 트레이드오프)\n');
fprintf('   - FFT 기반으로 계산 효율적\n\n');

fprintf('2. 필터뱅크 방식:\n');
fprintf('   - 최적 사용 조건: 실시간 처리, 특정 주파수 대역 모니터링\n');
fprintf('   - 각 대역 독립적 처리 가능\n');
fprintf('   - 필터 설계가 성능에 큰 영향\n\n');

fprintf('3. 선택 가이드:\n');
fprintf('   - 오프라인 분석 → STFT\n');
fprintf('   - 실시간 처리 → 필터뱅크\n');
fprintf('   - 높은 주파수 해상도 필요 → STFT (긴 윈도우)\n');
fprintf('   - 높은 시간 해상도 필요 → 필터뱅크 또는 STFT (짧은 윈도우)\n');