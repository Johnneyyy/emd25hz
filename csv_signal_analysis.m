%% CSV 파일 기반 신호 분석: STFT vs 필터뱅크
% CSV 파일에서 신호를 로드하여 STFT와 필터뱅크로 분석
clear all; close all; clc;

%% 1. CSV 파일 로드
fprintf('=== CSV 파일 기반 신호 분석 ===\n\n');

% CSV 파일 선택 (사용자가 파일 경로를 지정할 수 있음)
csv_file = 'signal_data.csv';  % 기본 파일명

% 파일 존재 여부 확인
if ~exist(csv_file, 'file')
    fprintf('CSV 파일 (%s)이 존재하지 않습니다.\n', csv_file);
    fprintf('샘플 CSV 파일을 생성하시겠습니까? (y/n): ');
    answer = input('', 's');
    
    if strcmpi(answer, 'y')
        % 샘플 CSV 파일 생성
        create_sample_csv(csv_file);
        fprintf('샘플 CSV 파일이 생성되었습니다: %s\n\n', csv_file);
    else
        error('CSV 파일이 필요합니다.');
    end
end

% CSV 파일 로드
fprintf('CSV 파일 로드 중: %s\n', csv_file);
data = readmatrix(csv_file);

% 데이터 구조 확인
[n_samples, n_columns] = size(data);
fprintf('데이터 크기: %d 샘플 x %d 열\n', n_samples, n_columns);

% 시간과 신호 분리 (첫 번째 열: 시간, 두 번째 열: 신호)
if n_columns >= 2
    t = data(:, 1)';
    signal = data(:, 2)';
    fs = round(1/mean(diff(t)));  % 샘플링 주파수 추정
    fprintf('추정된 샘플링 주파수: %d Hz\n', fs);
else
    % 시간 정보가 없는 경우
    signal = data(:, 1)';
    fs = input('샘플링 주파수를 입력하세요 (Hz): ');
    t = (0:length(signal)-1) / fs;
end

% 신호 정보 출력
fprintf('\n신호 정보:\n');
fprintf('  - 길이: %.2f 초\n', t(end));
fprintf('  - 샘플 수: %d\n', length(signal));
fprintf('  - 최대값: %.4f\n', max(signal));
fprintf('  - 최소값: %.4f\n', min(signal));
fprintf('  - 평균값: %.4f\n', mean(signal));
fprintf('  - 표준편차: %.4f\n\n', std(signal));

%% 2. 분석 파라미터 설정
% 주파수 대역 설정
nyquist_freq = fs/2;
freq_start = 0.1;
freq_end = min(nyquist_freq * 0.8, 20);  % Nyquist 주파수의 80% 또는 20Hz 중 작은 값
freq_step = (freq_end - freq_start) / 20;  % 20개 대역
freq_bands = freq_start:freq_step:freq_end;
n_bands = length(freq_bands);

fprintf('분석 주파수 범위: %.2f Hz ~ %.2f Hz\n', freq_start, freq_end);
fprintf('주파수 대역 수: %d\n\n', n_bands);

%% 3. STFT 분석
fprintf('STFT 분석 수행 중...\n');

% STFT 파라미터
window_length = min(round(fs * 2), round(length(signal)/10));  % 2초 또는 신호 길이의 10%
overlap = round(window_length * 0.9);  % 90% 오버랩
nfft = 2^nextpow2(window_length * 2);

% STFT 수행
[S, F, T_stft] = spectrogram(signal, hamming(window_length), overlap, nfft, fs);

% 주파수 대역별 파워 계산
stft_power = zeros(n_bands, length(T_stft));
for i = 1:n_bands
    freq_idx = find(F >= (freq_bands(i)-freq_step/2) & F <= (freq_bands(i)+freq_step/2));
    if ~isempty(freq_idx)
        stft_power(i,:) = mean(abs(S(freq_idx,:)).^2, 1);
    end
end

fprintf('STFT 완료: 윈도우 크기 = %.2f초, 시간 해상도 = %.3f초\n', ...
    window_length/fs, mean(diff(T_stft)));

%% 4. 필터뱅크 분석
fprintf('필터뱅크 분석 수행 중...\n');

% 필터뱅크 파라미터
filter_order = 4;  % 버터워스 필터 차수

% 각 주파수 대역에 대한 필터링
filterbank_power = zeros(n_bands, length(t));
filterbank_phase = zeros(n_bands, length(t));

for i = 1:n_bands
    % 대역통과 필터 설계
    if i == 1
        % 첫 번째 대역
        f_low = max(0.01, freq_bands(i) - freq_step/2);
        f_high = freq_bands(i) + freq_step/2;
    elseif i == n_bands
        % 마지막 대역
        f_low = freq_bands(i) - freq_step/2;
        f_high = min(nyquist_freq * 0.95, freq_bands(i) + freq_step/2);
    else
        % 중간 대역
        f_low = freq_bands(i) - freq_step/2;
        f_high = freq_bands(i) + freq_step/2;
    end
    
    % 주파수 정규화 확인
    f_low_norm = f_low / nyquist_freq;
    f_high_norm = f_high / nyquist_freq;
    
    if f_low_norm <= 0 || f_high_norm >= 1 || f_low_norm >= f_high_norm
        fprintf('  대역 %d 스킵 (주파수 범위 오류)\n', i);
        continue;
    end
    
    try
        [b, a] = butter(filter_order, [f_low_norm, f_high_norm], 'bandpass');
        
        % 필터링 수행
        filtered_signal = filtfilt(b, a, signal);
        
        % 힐버트 변환으로 순시 진폭과 위상 계산
        analytic_signal = hilbert(filtered_signal);
        filterbank_power(i,:) = abs(analytic_signal).^2;
        filterbank_phase(i,:) = angle(analytic_signal);
    catch ME
        fprintf('  대역 %d 필터링 실패: %s\n', i, ME.message);
    end
    
    if mod(i, 5) == 0
        fprintf('  진행률: %d/%d 대역 완료\n', i, n_bands);
    end
end

fprintf('필터뱅크 분석 완료\n\n');

%% 5. 결과 시각화
figure('Position', [50, 50, 1600, 900]);
sgtitle('CSV 신호 분석: STFT vs 필터뱅크', 'FontSize', 14, 'FontWeight', 'bold');

% 5.1 원본 신호
subplot(3,3,1);
plot(t, signal, 'b-', 'LineWidth', 0.5);
xlabel('시간 (초)');
ylabel('진폭');
title('원본 신호');
grid on;
xlim([t(1), t(end)]);

% 5.2 신호 스펙트럼 (FFT)
subplot(3,3,2);
Y = fft(signal);
f_fft = fs*(0:(length(signal)/2))/length(signal);
P = abs(Y/length(signal));
P = P(1:length(signal)/2+1);
P(2:end-1) = 2*P(2:end-1);
plot(f_fft, 20*log10(P + eps), 'r-', 'LineWidth', 1);
xlabel('주파수 (Hz)');
ylabel('파워 (dB)');
title('주파수 스펙트럼 (FFT)');
grid on;
xlim([0, min(fs/2, 50)]);

% 5.3 신호 통계
subplot(3,3,3);
text(0.1, 0.9, '신호 통계:', 'FontWeight', 'bold', 'FontSize', 11);
text(0.1, 0.75, sprintf('평균: %.4f', mean(signal)));
text(0.1, 0.6, sprintf('표준편차: %.4f', std(signal)));
text(0.1, 0.45, sprintf('최대값: %.4f', max(signal)));
text(0.1, 0.3, sprintf('최소값: %.4f', min(signal)));
text(0.1, 0.15, sprintf('RMS: %.4f', sqrt(mean(signal.^2))));
axis off;

% 5.4 STFT 스펙트로그램
subplot(3,3,4);
imagesc(T_stft, freq_bands, 10*log10(stft_power + eps));
axis xy;
xlabel('시간 (초)');
ylabel('주파수 (Hz)');
title(sprintf('STFT 스펙트로그램 (윈도우: %.1f초)', window_length/fs));
colorbar;
colormap(jet);
caxis([-60, max(10*log10(stft_power(:) + eps))]);

% 5.5 필터뱅크 스펙트로그램
subplot(3,3,5);
% 다운샘플링 (표시 최적화)
downsample_factor = max(1, round(length(t)/1000));
t_down = t(1:downsample_factor:end);
fb_power_down = filterbank_power(:, 1:downsample_factor:end);
imagesc(t_down, freq_bands, 10*log10(fb_power_down + eps));
axis xy;
xlabel('시간 (초)');
ylabel('주파수 (Hz)');
title(sprintf('필터뱅크 스펙트로그램 (%d차 필터)', filter_order));
colorbar;
colormap(jet);
caxis([-60, max(10*log10(fb_power_down(:) + eps))]);

% 5.6 시간 해상도 비교
subplot(3,3,6);
% 특정 주파수 대역의 시간 변화 비교
band_idx = round(n_bands/2);  % 중간 주파수 대역
plot(T_stft, stft_power(band_idx,:)/max(stft_power(band_idx,:)), 'b-', 'LineWidth', 1.5);
hold on;
plot(t(1:downsample_factor:end), ...
    filterbank_power(band_idx,1:downsample_factor:end)/max(filterbank_power(band_idx,:)), ...
    'r--', 'LineWidth', 1.5);
xlabel('시간 (초)');
ylabel('정규화된 파워');
title(sprintf('%.2f Hz 대역 시간 변화', freq_bands(band_idx)));
legend('STFT', '필터뱅크', 'Location', 'best');
grid on;
xlim([t(1), t(end)]);

% 5.7 주파수 해상도 비교
subplot(3,3,7);
% 특정 시간의 주파수 스펙트럼
time_idx_stft = round(length(T_stft)/2);
time_idx_fb = round(length(t)/2);
plot(freq_bands, 10*log10(stft_power(:,time_idx_stft) + eps), 'b-o', 'LineWidth', 1.5);
hold on;
plot(freq_bands, 10*log10(filterbank_power(:,time_idx_fb) + eps), 'r--s', 'LineWidth', 1.5);
xlabel('주파수 (Hz)');
ylabel('파워 (dB)');
title(sprintf('t=%.1f초 주파수 스펙트럼', t(time_idx_fb)));
legend('STFT', '필터뱅크', 'Location', 'best');
grid on;

% 5.8 위상 정보 (필터뱅크)
subplot(3,3,8);
imagesc(t_down, freq_bands, filterbank_phase(:, 1:downsample_factor:end));
axis xy;
xlabel('시간 (초)');
ylabel('주파수 (Hz)');
title('필터뱅크 위상 정보');
colorbar;
colormap(hsv);

% 5.9 성능 메트릭
subplot(3,3,9);
text(0.05, 0.95, '분석 성능 비교:', 'FontWeight', 'bold', 'FontSize', 11);
text(0.05, 0.80, 'STFT:', 'FontWeight', 'bold');
text(0.1, 0.70, sprintf('- 시간 해상도: %.3f초', mean(diff(T_stft))));
text(0.1, 0.60, sprintf('- 주파수 빈: %d', length(F)));
text(0.1, 0.50, sprintf('- 윈도우 크기: %d 샘플', window_length));

text(0.05, 0.35, '필터뱅크:', 'FontWeight', 'bold');
text(0.1, 0.25, sprintf('- 시간 해상도: %.3f초', 1/fs));
text(0.1, 0.15, sprintf('- 주파수 대역: %d', n_bands));
text(0.1, 0.05, sprintf('- 필터 차수: %d', filter_order));
axis off;

%% 6. 상세 분석 결과 출력
fprintf('=== 분석 결과 요약 ===\n\n');

% 주요 주파수 성분 찾기
[pks, locs] = findpeaks(P, 'MinPeakHeight', max(P)*0.1);
if ~isempty(locs)
    fprintf('주요 주파수 성분:\n');
    for i = 1:min(5, length(locs))
        fprintf('  %.2f Hz (파워: %.2f dB)\n', f_fft(locs(i)), 20*log10(pks(i)));
    end
else
    fprintf('주요 주파수 성분을 찾을 수 없습니다.\n');
end

fprintf('\nSTFT 분석:\n');
fprintf('  - 윈도우 크기: %.2f초 (%d 샘플)\n', window_length/fs, window_length);
fprintf('  - 오버랩: %.1f%%\n', (overlap/window_length)*100);
fprintf('  - FFT 포인트: %d\n', nfft);
fprintf('  - 시간 프레임 수: %d\n', length(T_stft));

fprintf('\n필터뱅크 분석:\n');
fprintf('  - 필터 유형: Butterworth\n');
fprintf('  - 필터 차수: %d\n', filter_order);
fprintf('  - 주파수 대역 수: %d\n', n_bands);
fprintf('  - 대역폭: %.2f Hz\n', freq_step);

%% 7. 결과 저장 옵션
fprintf('\n결과를 저장하시겠습니까? (y/n): ');
save_answer = input('', 's');

if strcmpi(save_answer, 'y')
    % 결과 저장
    [~, name, ~] = fileparts(csv_file);
    output_dir = 'analysis_results';
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end
    
    % STFT 결과 저장
    stft_file = fullfile(output_dir, [name '_stft_result.mat']);
    save(stft_file, 'T_stft', 'freq_bands', 'stft_power', 'fs');
    fprintf('STFT 결과 저장: %s\n', stft_file);
    
    % 필터뱅크 결과 저장
    fb_file = fullfile(output_dir, [name '_filterbank_result.mat']);
    save(fb_file, 't', 'freq_bands', 'filterbank_power', 'filterbank_phase', 'fs');
    fprintf('필터뱅크 결과 저장: %s\n', fb_file);
    
    % 그래프 저장
    fig_file = fullfile(output_dir, [name '_analysis_plot.png']);
    saveas(gcf, fig_file);
    fprintf('그래프 저장: %s\n', fig_file);
end

%% 샘플 CSV 파일 생성 함수
function create_sample_csv(filename)
    % 샘플 신호 생성
    fs = 100;  % 샘플링 주파수
    duration = 30;  % 30초
    t = (0:1/fs:duration-1/fs)';
    
    % 복합 신호 생성 (여러 주파수 성분 포함)
    signal = 0.5*sin(2*pi*0.5*t) + ...     % 0.5 Hz
             0.3*sin(2*pi*1.2*t) + ...     % 1.2 Hz
             0.4*sin(2*pi*2.5*t) + ...     % 2.5 Hz
             0.2*sin(2*pi*5.0*t) + ...     % 5.0 Hz
             0.1*randn(size(t));           % 잡음
    
    % 시변 성분 추가
    for i = 1:length(t)
        if t(i) > 10 && t(i) < 20
            signal(i) = signal(i) + 0.3*sin(2*pi*8.0*t(i));
        end
    end
    
    % CSV 파일로 저장
    data = [t, signal];
    writematrix(data, filename);
    
    fprintf('샘플 데이터 생성:\n');
    fprintf('  - 샘플링 주파수: %d Hz\n', fs);
    fprintf('  - 신호 길이: %d 초\n', duration);
    fprintf('  - 포함된 주파수: 0.5, 1.2, 2.5, 5.0 Hz\n');
    fprintf('  - 시변 성분: 10-20초 구간에 8.0 Hz\n');
end