%% 실시간 시뮬레이션: STFT vs 필터뱅크
% 실시간 처리 상황에서의 두 방법 비교
clear all; close all; clc;

%% 1. 시뮬레이션 파라미터
fs = 100;  % 샘플링 주파수 (Hz)
buffer_size = fs * 1;  % 1초 버퍼
total_duration = 30;  % 전체 시뮬레이션 시간 (초)
n_buffers = total_duration;

% 주파수 대역 설정
freq_bands = 0.2:0.2:4.0;  % 19개 대역
n_bands = length(freq_bands);

% 실시간으로 변하는 신호 생성 함수
generate_signal = @(t, idx) ...
    0.5*sin(2*pi*(0.5 + 0.1*sin(2*pi*0.05*idx))*t) + ...  % 주파수 변조
    0.3*sin(2*pi*(2.0 + idx/30)*t) + ...                   % 선형 주파수 증가
    0.2*(rand(1)-0.5)*sin(2*pi*3.5*t) + ...               % 진폭 변조
    0.1*randn(size(t));                                    % 잡음

%% 2. 필터뱅크 초기화
% 각 대역의 필터 계수 미리 계산
filter_bank = cell(n_bands, 1);
for i = 1:n_bands
    if i == 1
        [b, a] = butter(4, [0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    elseif i == n_bands
        [b, a] = butter(4, [freq_bands(i)-0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    else
        [b, a] = butter(4, [freq_bands(i)-0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    end
    filter_bank{i} = {b, a};
end

% 필터 상태 초기화
filter_states = cell(n_bands, 1);
for i = 1:n_bands
    filter_states{i} = [];
end

%% 3. STFT 파라미터
stft_window_size = fs * 5;  % 5초 윈도우
stft_overlap = stft_window_size * 0.9;  % 90% 오버랩
stft_buffer = [];  % STFT용 버퍼
stft_hop = stft_window_size - stft_overlap;

%% 4. 결과 저장 변수
fb_results = zeros(n_bands, n_buffers);
stft_results = [];
fb_times = zeros(n_buffers, 1);
stft_times = zeros(n_buffers, 1);

%% 5. 실시간 시뮬레이션
figure('Position', [100, 100, 1400, 800]);

for buf_idx = 1:n_buffers
    % 새로운 데이터 생성
    t_current = ((buf_idx-1)*buffer_size:(buf_idx*buffer_size-1))/fs;
    new_data = generate_signal(t_current, buf_idx);
    
    %% 5.1 필터뱅크 처리
    tic;
    fb_power = zeros(n_bands, 1);
    for i = 1:n_bands
        b = filter_bank{i}{1};
        a = filter_bank{i}{2};
        
        % 상태를 유지하면서 필터링
        if isempty(filter_states{i})
            [filtered, filter_states{i}] = filter(b, a, new_data);
        else
            [filtered, filter_states{i}] = filter(b, a, new_data, filter_states{i});
        end
        
        % 순시 파워 계산
        analytic = hilbert(filtered);
        fb_power(i) = mean(abs(analytic).^2);
    end
    fb_results(:, buf_idx) = fb_power;
    fb_times(buf_idx) = toc;
    
    %% 5.2 STFT 처리
    tic;
    % 버퍼에 새 데이터 추가
    stft_buffer = [stft_buffer, new_data];
    
    % 충분한 데이터가 모이면 STFT 수행
    if length(stft_buffer) >= stft_window_size
        % STFT 계산
        window = hamming(stft_window_size);
        nfft = 2^nextpow2(stft_window_size);
        
        % 윈도우 적용 및 FFT
        windowed_data = stft_buffer(end-stft_window_size+1:end) .* window';
        spectrum = fft(windowed_data, nfft);
        spectrum = spectrum(1:nfft/2+1);
        psd = abs(spectrum).^2 / (fs * sum(window.^2));
        
        % 주파수 빈
        f = fs * (0:nfft/2) / nfft;
        
        % 각 대역의 파워 계산
        stft_power = zeros(n_bands, 1);
        for i = 1:n_bands
            freq_idx = find(f >= (freq_bands(i)-0.1) & f <= (freq_bands(i)+0.1));
            if ~isempty(freq_idx)
                stft_power(i) = mean(psd(freq_idx));
            end
        end
        
        stft_results = [stft_results, stft_power];
        
        % 버퍼 업데이트 (hop size만큼 이동)
        if length(stft_buffer) > stft_window_size
            stft_buffer = stft_buffer(stft_hop+1:end);
        end
    end
    stft_times(buf_idx) = toc;
    
    %% 5.3 실시간 시각화 (5번마다 업데이트)
    if mod(buf_idx, 5) == 0
        % 필터뱅크 결과
        subplot(2,2,1);
        imagesc(1:buf_idx, freq_bands, 10*log10(fb_results(:,1:buf_idx) + eps));
        axis xy;
        xlabel('시간 (초)');
        ylabel('주파수 (Hz)');
        title('필터뱅크 실시간 결과');
        colorbar;
        caxis([-40, 0]);
        
        % STFT 결과
        if ~isempty(stft_results)
            subplot(2,2,2);
            imagesc(1:size(stft_results,2), freq_bands, 10*log10(stft_results + eps));
            axis xy;
            xlabel('STFT 프레임');
            ylabel('주파수 (Hz)');
            title('STFT 실시간 결과');
            colorbar;
            caxis([-40, 0]);
        end
        
        % 처리 시간 비교
        subplot(2,2,3);
        plot(1:buf_idx, fb_times(1:buf_idx)*1000, 'b-', 'LineWidth', 2);
        hold on;
        plot(1:buf_idx, stft_times(1:buf_idx)*1000, 'r--', 'LineWidth', 2);
        hold off;
        xlabel('버퍼 인덱스');
        ylabel('처리 시간 (ms)');
        title('실시간 처리 시간');
        legend('필터뱅크', 'STFT', 'Location', 'best');
        grid on;
        ylim([0, max([fb_times; stft_times])*1200]);
        
        % 현재 스펙트럼
        subplot(2,2,4);
        if buf_idx > 5
            recent_fb = mean(fb_results(:, buf_idx-4:buf_idx), 2);
            plot(freq_bands, 10*log10(recent_fb + eps), 'b-o', 'LineWidth', 2);
            if size(stft_results, 2) > 0
                hold on;
                recent_stft = stft_results(:, end);
                plot(freq_bands, 10*log10(recent_stft + eps), 'r--s', 'LineWidth', 2);
                hold off;
                legend('필터뱅크', 'STFT', 'Location', 'best');
            end
        end
        xlabel('주파수 (Hz)');
        ylabel('파워 (dB)');
        title('현재 스펙트럼 (최근 5초 평균)');
        grid on;
        ylim([-40, 0]);
        
        drawnow;
    end
end

%% 6. 최종 성능 분석
fprintf('\n=== 실시간 처리 성능 분석 ===\n');
fprintf('평균 처리 시간:\n');
fprintf('  - 필터뱅크: %.2f ms (±%.2f ms)\n', mean(fb_times)*1000, std(fb_times)*1000);
fprintf('  - STFT: %.2f ms (±%.2f ms)\n', mean(stft_times)*1000, std(stft_times)*1000);

fprintf('\n최대 처리 시간:\n');
fprintf('  - 필터뱅크: %.2f ms\n', max(fb_times)*1000);
fprintf('  - STFT: %.2f ms\n', max(stft_times)*1000);

fprintf('\n실시간 처리 가능성 (1초 버퍼 기준):\n');
fprintf('  - 필터뱅크: %s\n', iff(max(fb_times) < 1, '가능', '불가능'));
fprintf('  - STFT: %s\n', iff(max(stft_times) < 1, '가능', '불가능'));

% 지연시간 분석
fprintf('\n지연시간 분석:\n');
fprintf('  - 필터뱅크: 최소 지연 (버퍼 크기: %d 샘플 = %.0f ms)\n', ...
    buffer_size, buffer_size/fs*1000);
fprintf('  - STFT: 윈도우 크기에 따른 지연 (%d 샘플 = %.0f ms)\n', ...
    stft_window_size, stft_window_size/fs*1000);

%% 7. 추가 분석 그래프
figure('Position', [100, 100, 1200, 600]);

% 7.1 처리 시간 히스토그램
subplot(1,2,1);
histogram(fb_times*1000, 20, 'FaceColor', 'b', 'FaceAlpha', 0.5);
hold on;
histogram(stft_times*1000, 20, 'FaceColor', 'r', 'FaceAlpha', 0.5);
xlabel('처리 시간 (ms)');
ylabel('빈도');
title('처리 시간 분포');
legend('필터뱅크', 'STFT');
grid on;

% 7.2 누적 처리 시간
subplot(1,2,2);
plot(1:n_buffers, cumsum(fb_times), 'b-', 'LineWidth', 2);
hold on;
plot(1:n_buffers, cumsum(stft_times), 'r--', 'LineWidth', 2);
plot(1:n_buffers, (1:n_buffers)', 'k:', 'LineWidth', 1);
xlabel('버퍼 인덱스');
ylabel('누적 처리 시간 (초)');
title('누적 처리 시간 vs 실제 시간');
legend('필터뱅크', 'STFT', '실시간 기준선', 'Location', 'northwest');
grid on;

%% Helper function
function result = iff(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end