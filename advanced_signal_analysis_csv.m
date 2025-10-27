%% 고급 신호 분석: STFT vs 필터뱅크 상세 비교 (CSV 입력)
% CSV 파일에서 신호를 읽어 다양한 분석 수행
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

%% 2. 분석 파라미터 설정
freq_start = 0.2;
freq_end = 4.0;
freq_step = 0.2;
freq_bands = freq_start:freq_step:freq_end;
n_bands = length(freq_bands);

% STFT 파라미터 (신호 길이에 따라 적응적으로 설정)
if duration < 30
    window_lengths = [2*fs, 5*fs];  % 짧은 신호
    window_names = {'2초', '5초'};
elseif duration < 120
    window_lengths = [5*fs, 10*fs, 15*fs];  % 중간 신호
    window_names = {'5초', '10초', '15초'};
else
    window_lengths = [10*fs, 20*fs, 30*fs];  % 긴 신호
    window_names = {'10초', '20초', '30초'};
end

% 실제 사용 가능한 윈도우 크기로 조정
window_lengths = window_lengths(window_lengths < length(signal)/2);
window_names = window_names(1:length(window_lengths));

%% 3. 신호 특성 분석
figure('Position', [100, 100, 1600, 900]);
sgtitle(sprintf('신호 특성 분석: %s', filename), 'Interpreter', 'none');

% 3.1 원본 신호 표시
subplot(3,3,1);
plot(t, signal);
xlabel('시간 (초)');
ylabel('진폭');
title('원본 신호');
grid on;
xlim([t(1), t(end)]);

% 3.2 신호 통계
subplot(3,3,2);
text(0.1, 0.9, sprintf('평균: %.4f', mean(signal)), 'FontSize', 10);
text(0.1, 0.8, sprintf('표준편차: %.4f', std(signal)), 'FontSize', 10);
text(0.1, 0.7, sprintf('최대값: %.4f', max(signal)), 'FontSize', 10);
text(0.1, 0.6, sprintf('최소값: %.4f', min(signal)), 'FontSize', 10);
text(0.1, 0.5, sprintf('RMS: %.4f', rms(signal)), 'FontSize', 10);
text(0.1, 0.4, sprintf('첨도: %.4f', kurtosis(signal)), 'FontSize', 10);
text(0.1, 0.3, sprintf('왜도: %.4f', skewness(signal)), 'FontSize', 10);
axis off;
title('신호 통계');

% 3.3 전체 스펙트럼
subplot(3,3,3);
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

% 3.4 다양한 윈도우 크기의 STFT
for win_idx = 1:length(window_lengths)
    window_length = window_lengths(win_idx);
    overlap = round(window_length * 0.9);
    nfft = 2^nextpow2(window_length * 4);
    
    [S, F, T_stft] = spectrogram(signal, hamming(window_length), overlap, nfft, fs);
    
    % 주파수 대역별 파워 추출
    stft_power = zeros(n_bands, length(T_stft));
    for i = 1:n_bands
        freq_idx = find(F >= (freq_bands(i)-0.1) & F <= (freq_bands(i)+0.1));
        if ~isempty(freq_idx)
            stft_power(i,:) = mean(abs(S(freq_idx,:)).^2, 1);
        end
    end
    
    subplot(3,3,3+win_idx);
    imagesc(T_stft, freq_bands, 10*log10(stft_power + eps));
    axis xy;
    xlabel('시간 (초)');
    ylabel('주파수 (Hz)');
    title(sprintf('STFT (%s 윈도우)', window_names{win_idx}));
    colorbar;
    caxis([-60, max(10*log10(stft_power(:)))]);
end

% 3.5 다양한 필터 차수의 필터뱅크
filter_orders = [2, 4, 6];

for ord_idx = 1:length(filter_orders)
    filter_order = filter_orders(ord_idx);
    filterbank_power = zeros(n_bands, length(t));
    
    for i = 1:n_bands
        if i == 1
            [b, a] = butter(filter_order, [0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
        elseif i == n_bands
            [b, a] = butter(filter_order, [freq_bands(i)-0.1, min(freq_bands(i)+0.1, fs/2-0.1)]/(fs/2), 'bandpass');
        else
            [b, a] = butter(filter_order, [freq_bands(i)-0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
        end
        
        filtered_signal = filtfilt(b, a, signal);
        analytic_signal = hilbert(filtered_signal);
        filterbank_power(i,:) = abs(analytic_signal).^2;
    end
    
    % 다운샘플링 (표시용)
    downsample_factor = max(1, round(length(t) / 1000));  % 최대 1000 포인트
    t_down = t(1:downsample_factor:end);
    fb_power_down = filterbank_power(:, 1:downsample_factor:end);
    
    subplot(3,3,6+ord_idx);
    imagesc(t_down, freq_bands, 10*log10(fb_power_down + eps));
    axis xy;
    xlabel('시간 (초)');
    ylabel('주파수 (Hz)');
    title(sprintf('필터뱅크 (%d차)', filter_order));
    colorbar;
    caxis([-60, max(10*log10(fb_power_down(:)))]);
end

%% 4. 정량적 성능 비교
figure('Position', [100, 100, 1200, 800]);
sgtitle('성능 메트릭 비교', 'FontSize', 14);

% 4.1 시간-주파수 불확정성 원리 시각화
subplot(2,2,1);
window_sizes_sec = window_lengths / fs;  % 초 단위
freq_resolution = fs ./ window_lengths;  % Hz
time_resolution = window_sizes_sec;  % 초

plot(time_resolution, freq_resolution, 'bo-', 'LineWidth', 2);
xlabel('시간 해상도 (초)');
ylabel('주파수 해상도 (Hz)');
title('시간-주파수 해상도 트레이드오프');
grid on;
text(mean(time_resolution), mean(freq_resolution)*1.2, ...
    '불확정성 원리: Δt × Δf ≥ 1/(4π)', 'FontSize', 10, 'HorizontalAlignment', 'center');

% 4.2 필터뱅크 주파수 응답
subplot(2,2,2);
freqz_points = 1024;
colors = lines(length(filter_orders));

for ord_idx = 1:length(filter_orders)
    filter_order = filter_orders(ord_idx);
    hold on;
    
    % 몇 개의 대표 필터만 표시
    representative_bands = [1, 5, 10, 15, 19];
    for band_idx = representative_bands
        if band_idx <= n_bands
            i = band_idx;
            if i == 1
                [b, a] = butter(filter_order, [0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
            elseif i == n_bands
                [b, a] = butter(filter_order, [freq_bands(i)-0.1, min(freq_bands(i)+0.1, fs/2-0.1)]/(fs/2), 'bandpass');
            else
                [b, a] = butter(filter_order, [freq_bands(i)-0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
            end
            [H, W] = freqz(b, a, freqz_points);
            f_freqz = W * fs / (2*pi);
            
            if band_idx == representative_bands(1)
                plot(f_freqz, 20*log10(abs(H)), 'Color', colors(ord_idx,:), ...
                    'LineWidth', 1.5, 'DisplayName', sprintf('%d차 필터', filter_order));
            else
                plot(f_freqz, 20*log10(abs(H)), 'Color', colors(ord_idx,:), ...
                    'LineWidth', 1.5, 'HandleVisibility', 'off');
            end
        end
    end
end
xlabel('주파수 (Hz)');
ylabel('크기 응답 (dB)');
title('필터뱅크 주파수 응답 (대표 필터)');
legend('Location', 'southwest');
grid on;
xlim([0, 5]);
ylim([-60, 5]);

% 4.3 계산 시간 비교
subplot(2,2,3);
% 다양한 길이의 신호에 대한 계산 시간 측정
test_lengths = round(linspace(length(signal)/10, length(signal), 5));
stft_times = zeros(length(window_lengths), length(test_lengths));
fb_times = zeros(length(filter_orders), length(test_lengths));

for len_idx = 1:length(test_lengths)
    test_signal = signal(1:test_lengths(len_idx));
    
    % STFT 시간
    for win_idx = 1:length(window_lengths)
        if window_lengths(win_idx) < test_lengths(len_idx)/2
            tic;
            spectrogram(test_signal, hamming(window_lengths(win_idx)), ...
                round(window_lengths(win_idx)*0.9), 2048, fs);
            stft_times(win_idx, len_idx) = toc;
        end
    end
    
    % 필터뱅크 시간
    for ord_idx = 1:length(filter_orders)
        tic;
        for j = 1:n_bands
            [b, a] = butter(filter_orders(ord_idx), ...
                [max(0.1, freq_bands(j)-0.1), min(freq_bands(j)+0.1, fs/2-0.1)]/(fs/2), 'bandpass');
            filtered = filtfilt(b, a, test_signal);
            hilbert(filtered);
        end
        fb_times(ord_idx, len_idx) = toc;
    end
end

test_durations = test_lengths / fs;
plot(test_durations, mean(stft_times, 1), 'b-o', 'LineWidth', 2, 'DisplayName', 'STFT (평균)');
hold on;
plot(test_durations, mean(fb_times, 1), 'r--s', 'LineWidth', 2, 'DisplayName', '필터뱅크 (평균)');
xlabel('신호 길이 (초)');
ylabel('계산 시간 (초)');
title('계산 시간 비교');
legend('Location', 'northwest');
grid on;

% 4.4 주파수별 에너지 분포
subplot(2,2,4);
% 최적 파라미터로 재계산
optimal_window = window_lengths(min(2, length(window_lengths)));
[S, F, T_opt] = spectrogram(signal, hamming(optimal_window), ...
    round(optimal_window*0.9), 2^nextpow2(optimal_window*4), fs);

stft_power_opt = zeros(n_bands, length(T_opt));
for i = 1:n_bands
    freq_idx = find(F >= (freq_bands(i)-0.1) & F <= (freq_bands(i)+0.1));
    if ~isempty(freq_idx)
        stft_power_opt(i,:) = mean(abs(S(freq_idx,:)).^2, 1);
    end
end

% 필터뱅크 (4차)
filterbank_power_opt = zeros(n_bands, length(t));
for i = 1:n_bands
    if i == 1
        [b, a] = butter(4, [0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    elseif i == n_bands
        [b, a] = butter(4, [freq_bands(i)-0.1, min(freq_bands(i)+0.1, fs/2-0.1)]/(fs/2), 'bandpass');
    else
        [b, a] = butter(4, [freq_bands(i)-0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    end
    
    filtered_signal = filtfilt(b, a, signal);
    analytic_signal = hilbert(filtered_signal);
    filterbank_power_opt(i,:) = abs(analytic_signal).^2;
end

mean_stft = mean(stft_power_opt, 2);
mean_fb = mean(filterbank_power_opt, 2);

plot(freq_bands, 10*log10(mean_stft + eps), 'b-o', 'LineWidth', 2);
hold on;
plot(freq_bands, 10*log10(mean_fb + eps), 'r--s', 'LineWidth', 2);
xlabel('주파수 (Hz)');
ylabel('평균 파워 (dB)');
title('주파수별 평균 파워 분포');
legend('STFT', '필터뱅크', 'Location', 'best');
grid on;

%% 5. 시간-주파수 특성 상세 분석
figure('Position', [100, 100, 1400, 800]);
sgtitle('시간-주파수 특성 상세 분석', 'FontSize', 14);

% 5.1 신호의 시간적 변화 감지
% 이동 평균과 표준편차 계산
window_size = round(fs);  % 1초 윈도우
signal_envelope = movmean(abs(signal), window_size);
signal_std = movstd(signal, window_size);

subplot(3,2,1);
plot(t, signal, 'b-', 'LineWidth', 0.5);
hold on;
plot(t, signal_envelope, 'r-', 'LineWidth', 2);
plot(t, signal_envelope + 2*signal_std, 'r--', 'LineWidth', 1);
plot(t, signal_envelope - 2*signal_std, 'r--', 'LineWidth', 1);
xlabel('시간 (초)');
ylabel('진폭');
title('신호 엔벨로프 및 변동성');
legend('원본 신호', '엔벨로프', '±2σ', 'Location', 'best');
grid on;

% 5.2 순시 주파수 추정
analytic_full = hilbert(signal);
inst_phase = unwrap(angle(analytic_full));
inst_freq = diff(inst_phase) * fs / (2*pi);
inst_freq = [inst_freq, inst_freq(end)];  % 길이 맞추기

subplot(3,2,2);
plot(t, inst_freq);
xlabel('시간 (초)');
ylabel('순시 주파수 (Hz)');
title('순시 주파수 추정');
ylim([0, 10]);
grid on;

% 5.3 STFT vs 필터뱅크 상관관계 히트맵
% 시간 축 맞추기
downsample_factor = round(length(t) / length(T_opt));
fb_downsampled = zeros(n_bands, length(T_opt));
for i = 1:n_bands
    if downsample_factor > 1
        fb_downsampled(i,:) = decimate(filterbank_power_opt(i,:), downsample_factor);
    else
        fb_downsampled(i,:) = interp1(1:length(filterbank_power_opt(i,:)), ...
            filterbank_power_opt(i,:), linspace(1, length(filterbank_power_opt(i,:)), length(T_opt)));
    end
end

correlation_matrix = zeros(n_bands, 1);
for i = 1:n_bands
    if ~any(isnan(stft_power_opt(i,:))) && ~any(isnan(fb_downsampled(i,:)))
        correlation_matrix(i) = corr(stft_power_opt(i,:)', fb_downsampled(i,:)');
    end
end

subplot(3,2,3);
bar(freq_bands, correlation_matrix);
xlabel('주파수 (Hz)');
ylabel('상관계수');
title('STFT-필터뱅크 상관계수');
ylim([0, 1]);
grid on;

% 5.4 스펙트럼 중심 주파수
spectral_centroid_stft = sum(freq_bands' .* mean_stft) / sum(mean_stft);
spectral_centroid_fb = sum(freq_bands' .* mean_fb) / sum(mean_fb);

subplot(3,2,4);
text(0.1, 0.8, '스펙트럼 특성:', 'FontWeight', 'bold', 'FontSize', 12);
text(0.1, 0.6, sprintf('STFT 중심 주파수: %.2f Hz', spectral_centroid_stft), 'FontSize', 10);
text(0.1, 0.5, sprintf('필터뱅크 중심 주파수: %.2f Hz', spectral_centroid_fb), 'FontSize', 10);
text(0.1, 0.3, sprintf('평균 상관계수: %.3f', mean(correlation_matrix(~isnan(correlation_matrix)))), 'FontSize', 10);
text(0.1, 0.2, sprintf('최소 상관계수: %.3f', min(correlation_matrix(~isnan(correlation_matrix)))), 'FontSize', 10);
text(0.1, 0.1, sprintf('최대 상관계수: %.3f', max(correlation_matrix(~isnan(correlation_matrix)))), 'FontSize', 10);
axis off;
title('분석 요약');

% 5.5 시간별 주요 주파수 추적
[~, max_freq_idx_stft] = max(stft_power_opt, [], 1);
[~, max_freq_idx_fb] = max(fb_downsampled, [], 1);

subplot(3,2,5);
plot(T_opt, freq_bands(max_freq_idx_stft), 'b-', 'LineWidth', 2);
hold on;
plot(T_opt, freq_bands(max_freq_idx_fb), 'r--', 'LineWidth', 2);
xlabel('시간 (초)');
ylabel('주파수 (Hz)');
title('시간별 주요 주파수 추적');
legend('STFT', '필터뱅크', 'Location', 'best');
ylim([0, 4.5]);
grid on;

% 5.6 에너지 시간 프로파일
total_energy_stft = sum(stft_power_opt, 1);
total_energy_fb = sum(fb_downsampled, 1);

subplot(3,2,6);
plot(T_opt, 10*log10(total_energy_stft + eps), 'b-', 'LineWidth', 2);
hold on;
plot(T_opt, 10*log10(total_energy_fb + eps), 'r--', 'LineWidth', 2);
xlabel('시간 (초)');
ylabel('총 에너지 (dB)');
title('시간별 총 에너지');
legend('STFT', '필터뱅크', 'Location', 'best');
grid on;

%% 6. 최종 요약 및 권장사항
fprintf('\n=== 최종 분석 요약 ===\n');
fprintf('1. 신호 특성:\n');
fprintf('   - 주요 주파수 성분: ');
[sorted_power, sorted_idx] = sort(mean_stft, 'descend');
top_freqs = freq_bands(sorted_idx(1:min(3, length(sorted_idx))));
fprintf('%.1f Hz, %.1f Hz, %.1f Hz\n', top_freqs(1), top_freqs(2), top_freqs(3));
fprintf('   - 스펙트럼 중심: %.2f Hz\n', spectral_centroid_stft);
fprintf('   - 신호 변동성: %.2f%%\n', std(signal)/mean(abs(signal))*100);

fprintf('\n2. STFT 분석:\n');
fprintf('   - 최적 윈도우 크기: %.1f초 (주파수 해상도: %.3f Hz)\n', ...
    optimal_window/fs, fs/optimal_window);
fprintf('   - 시간 해상도: %.3f초\n', (optimal_window - round(optimal_window*0.9))/fs);
fprintf('   - 계산 시간: %.3f초\n', mean(stft_times(:)));

fprintf('\n3. 필터뱅크 분석:\n');
fprintf('   - 최적 필터 차수: 4차 (경험적)\n');
fprintf('   - 대역폭: 0.2 Hz (고정)\n');
fprintf('   - 계산 시간: %.3f초\n', mean(fb_times(:)));

fprintf('\n4. 방법 선택 가이드:\n');
if duration < 30
    fprintf('   - 짧은 신호 (%.1f초): 필터뱅크 권장 (빠른 응답)\n', duration);
elseif duration < 120
    fprintf('   - 중간 신호 (%.1f초): 용도에 따라 선택\n', duration);
else
    fprintf('   - 긴 신호 (%.1f초): STFT 권장 (효율적)\n', duration);
end

fprintf('\n5. 두 방법의 일치도: %.1f%%\n', mean(correlation_matrix(~isnan(correlation_matrix)))*100);

%% 7. 결과 저장 옵션
save_results = questdlg('상세 분석 결과를 저장하시겠습니까?', ...
    '결과 저장', '예', '아니오', '예');

if strcmp(save_results, '예')
    % 분석 보고서 생성
    report_filename = strrep(filename, '.csv', '_analysis_report.txt');
    fid = fopen(report_filename, 'w');
    
    fprintf(fid, '신호 분석 보고서\n');
    fprintf(fid, '================\n\n');
    fprintf(fid, '파일명: %s\n', filename);
    fprintf(fid, '분석 일시: %s\n\n', datestr(now));
    
    fprintf(fid, '신호 정보:\n');
    fprintf(fid, '- 길이: %.2f 초\n', duration);
    fprintf(fid, '- 샘플 수: %d\n', length(signal));
    fprintf(fid, '- 샘플링 주파수: %d Hz\n\n', fs);
    
    fprintf(fid, '주요 주파수 성분:\n');
    for i = 1:min(5, length(sorted_idx))
        fprintf(fid, '- %.1f Hz: %.2f dB\n', ...
            freq_bands(sorted_idx(i)), 10*log10(sorted_power(i) + eps));
    end
    
    fprintf(fid, '\n상관계수 (STFT vs 필터뱅크):\n');
    fprintf(fid, '- 평균: %.3f\n', mean(correlation_matrix(~isnan(correlation_matrix))));
    fprintf(fid, '- 최소: %.3f\n', min(correlation_matrix(~isnan(correlation_matrix))));
    fprintf(fid, '- 최대: %.3f\n', max(correlation_matrix(~isnan(correlation_matrix))));
    
    fclose(fid);
    fprintf('\n분석 보고서가 저장되었습니다: %s\n', report_filename);
end