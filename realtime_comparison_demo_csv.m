%% 실시간 시뮬레이션: STFT vs 필터뱅크 (CSV 입력)
% CSV 파일의 신호를 실시간 스트리밍처럼 처리하여 비교
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

%% 2. 시뮬레이션 파라미터
buffer_size = fs * 1;  % 1초 버퍼
total_samples = length(signal);
n_buffers = floor(total_samples / buffer_size);

% 주파수 대역 설정
freq_bands = 0.2:0.2:4.0;  % 19개 대역
n_bands = length(freq_bands);

fprintf('\n=== 실시간 시뮬레이션 설정 ===\n');
fprintf('버퍼 크기: %d 샘플 (%.1f 초)\n', buffer_size, buffer_size/fs);
fprintf('총 버퍼 수: %d\n', n_buffers);
fprintf('시뮬레이션 시간: %.1f 초\n', n_buffers * buffer_size / fs);

%% 3. 필터뱅크 초기화
% 각 대역의 필터 계수 미리 계산
filter_bank = cell(n_bands, 1);
for i = 1:n_bands
    if i == 1
        [b, a] = butter(4, [0.1, freq_bands(i)+0.1]/(fs/2), 'bandpass');
    elseif i == n_bands
        [b, a] = butter(4, [freq_bands(i)-0.1, min(freq_bands(i)+0.1, fs/2-0.1)]/(fs/2), 'bandpass');
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

%% 4. STFT 파라미터
stft_window_size = min(fs * 5, buffer_size * 5);  % 5초 또는 5버퍼 중 작은 값
stft_overlap = round(stft_window_size * 0.9);  % 90% 오버랩
stft_buffer = [];  % STFT용 버퍼
stft_hop = stft_window_size - stft_overlap;

fprintf('\nSTFT 윈도우 크기: %d 샘플 (%.1f 초)\n', stft_window_size, stft_window_size/fs);

%% 5. 결과 저장 변수
fb_results = zeros(n_bands, n_buffers);
stft_results = [];
fb_times = zeros(n_buffers, 1);
stft_times = zeros(n_buffers, 1);
stft_frame_times = [];

%% 6. 실시간 시뮬레이션
figure('Position', [100, 100, 1400, 800]);
sgtitle(sprintf('실시간 처리 시뮬레이션: %s', filename), 'Interpreter', 'none');

fprintf('\n실시간 처리 시작...\n');
progress_interval = max(1, floor(n_buffers / 10));

for buf_idx = 1:n_buffers
    % 진행 상황 표시
    if mod(buf_idx, progress_interval) == 0
        fprintf('진행률: %d%% (%d/%d 버퍼)\n', round(buf_idx/n_buffers*100), buf_idx, n_buffers);
    end
    
    % 새로운 데이터 추출 (실제 신호에서)
    start_idx = (buf_idx-1) * buffer_size + 1;
    end_idx = min(buf_idx * buffer_size, total_samples);
    new_data = signal(start_idx:end_idx);
    
    % 마지막 버퍼가 불완전한 경우 패딩
    if length(new_data) < buffer_size
        new_data = [new_data, zeros(1, buffer_size - length(new_data))];
    end
    
    %% 6.1 필터뱅크 처리
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
    
    %% 6.2 STFT 처리
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
        stft_frame_times = [stft_frame_times, buf_idx];
        
        % 버퍼 업데이트 (hop size만큼 이동)
        if length(stft_buffer) > stft_window_size
            stft_buffer = stft_buffer(stft_hop+1:end);
        end
    end
    stft_times(buf_idx) = toc;
    
    %% 6.3 실시간 시각화 (5번마다 업데이트)
    if mod(buf_idx, 5) == 0 || buf_idx == n_buffers
        % 필터뱅크 결과
        subplot(2,2,1);
        imagesc((1:buf_idx)/fs, freq_bands, 10*log10(fb_results(:,1:buf_idx) + eps));
        axis xy;
        xlabel('시간 (초)');
        ylabel('주파수 (Hz)');
        title('필터뱅크 실시간 결과');
        colorbar;
        caxis([-60, max(10*log10(fb_results(:,1:buf_idx) + eps), [], 'all')]);
        
        % STFT 결과
        if ~isempty(stft_results)
            subplot(2,2,2);
            imagesc(stft_frame_times/fs, freq_bands, 10*log10(stft_results + eps));
            axis xy;
            xlabel('시간 (초)');
            ylabel('주파수 (Hz)');
            title('STFT 실시간 결과');
            colorbar;
            caxis([-60, max(10*log10(stft_results + eps), [], 'all')]);
        end
        
        % 처리 시간 비교
        subplot(2,2,3);
        plot((1:buf_idx)/fs, fb_times(1:buf_idx)*1000, 'b-', 'LineWidth', 2);
        hold on;
        plot((1:buf_idx)/fs, stft_times(1:buf_idx)*1000, 'r--', 'LineWidth', 2);
        
        % 실시간 처리 기준선 (1초 = 1000ms)
        yline(1000, 'k:', 'LineWidth', 2);
        hold off;
        
        xlabel('시간 (초)');
        ylabel('처리 시간 (ms)');
        title('실시간 처리 시간');
        legend('필터뱅크', 'STFT', '실시간 한계', 'Location', 'best');
        grid on;
        ylim([0, max([max(fb_times(1:buf_idx)), max(stft_times(1:buf_idx))])*1200]);
        
        % 현재 스펙트럼
        subplot(2,2,4);
        if buf_idx > 5
            recent_fb = mean(fb_results(:, max(1,buf_idx-4):buf_idx), 2);
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
        ylim([-60, 0]);
        
        drawnow;
    end
end

fprintf('실시간 처리 완료!\n');

%% 7. 최종 성능 분석
fprintf('\n=== 실시간 처리 성능 분석 ===\n');
fprintf('평균 처리 시간:\n');
fprintf('  - 필터뱅크: %.2f ms (±%.2f ms)\n', mean(fb_times)*1000, std(fb_times)*1000);
fprintf('  - STFT: %.2f ms (±%.2f ms)\n', mean(stft_times)*1000, std(stft_times)*1000);

fprintf('\n최대 처리 시간:\n');
fprintf('  - 필터뱅크: %.2f ms\n', max(fb_times)*1000);
fprintf('  - STFT: %.2f ms\n', max(stft_times)*1000);

fprintf('\n실시간 처리 가능성 (1초 버퍼 기준):\n');
fprintf('  - 필터뱅크: %s (%.1f%% 버퍼에서 가능)\n', ...
    iff(mean(fb_times) < 1, '가능', '평균적으로 불가능'), ...
    sum(fb_times < 1) / length(fb_times) * 100);
fprintf('  - STFT: %s (%.1f%% 버퍼에서 가능)\n', ...
    iff(mean(stft_times) < 1, '가능', '평균적으로 불가능'), ...
    sum(stft_times < 1) / length(stft_times) * 100);

% 지연시간 분석
fprintf('\n지연시간 분석:\n');
fprintf('  - 필터뱅크: 최소 지연 (버퍼 크기: %d 샘플 = %.0f ms)\n', ...
    buffer_size, buffer_size/fs*1000);
fprintf('  - STFT: 윈도우 크기에 따른 지연 (%d 샘플 = %.0f ms)\n', ...
    stft_window_size, stft_window_size/fs*1000);

%% 8. 추가 분석 그래프
figure('Position', [100, 100, 1200, 600]);
sgtitle('실시간 처리 성능 상세 분석', 'FontSize', 14);

% 8.1 처리 시간 히스토그램
subplot(1,3,1);
histogram(fb_times*1000, 20, 'FaceColor', 'b', 'FaceAlpha', 0.5);
hold on;
histogram(stft_times*1000, 20, 'FaceColor', 'r', 'FaceAlpha', 0.5);
xline(1000, 'k:', 'LineWidth', 2);
xlabel('처리 시간 (ms)');
ylabel('빈도');
title('처리 시간 분포');
legend('필터뱅크', 'STFT', '실시간 한계', 'Location', 'best');
grid on;

% 8.2 누적 처리 시간
subplot(1,3,2);
plot((1:n_buffers)/fs, cumsum(fb_times), 'b-', 'LineWidth', 2);
hold on;
plot((1:n_buffers)/fs, cumsum(stft_times), 'r--', 'LineWidth', 2);
plot((1:n_buffers)/fs, (1:n_buffers)', 'k:', 'LineWidth', 1);
xlabel('시간 (초)');
ylabel('누적 처리 시간 (초)');
title('누적 처리 시간 vs 실제 시간');
legend('필터뱅크', 'STFT', '실시간 기준선', 'Location', 'northwest');
grid on;

% 8.3 처리 효율성
subplot(1,3,3);
efficiency_fb = (1 ./ fb_times) / fs * buffer_size * 100;  % 처리 효율 (%)
efficiency_stft = (1 ./ stft_times) / fs * buffer_size * 100;

plot((1:n_buffers)/fs, efficiency_fb, 'b-', 'LineWidth', 1);
hold on;
plot((1:n_buffers)/fs, efficiency_stft, 'r--', 'LineWidth', 1);
yline(100, 'k:', 'LineWidth', 2);
xlabel('시간 (초)');
ylabel('처리 효율 (%)');
title('실시간 처리 효율');
legend('필터뱅크', 'STFT', '100% (실시간)', 'Location', 'best');
grid on;
ylim([0, max([max(efficiency_fb), max(efficiency_stft), 150])]);

%% 9. 결과 저장
save_results = questdlg('실시간 처리 결과를 저장하시겠습니까?', ...
    '결과 저장', '예', '아니오', '예');

if strcmp(save_results, '예')
    % 성능 메트릭 저장
    performance_filename = strrep(filename, '.csv', '_realtime_performance.csv');
    
    performance_data = [
        (1:n_buffers)'/fs, ...  % 시간
        fb_times*1000, ...      % 필터뱅크 처리 시간 (ms)
        stft_times*1000         % STFT 처리 시간 (ms)
    ];
    
    fid = fopen(performance_filename, 'w');
    fprintf(fid, 'Time(s),FilterBank_ms,STFT_ms\n');
    fclose(fid);
    dlmwrite(performance_filename, performance_data, '-append', 'delimiter', ',', 'precision', 6);
    
    fprintf('\n성능 데이터가 저장되었습니다: %s\n', performance_filename);
    
    % 요약 보고서
    summary_filename = strrep(filename, '.csv', '_realtime_summary.txt');
    fid = fopen(summary_filename, 'w');
    
    fprintf(fid, '실시간 처리 성능 요약\n');
    fprintf(fid, '===================\n\n');
    fprintf(fid, '파일명: %s\n', filename);
    fprintf(fid, '분석 일시: %s\n\n', datestr(now));
    
    fprintf(fid, '처리 설정:\n');
    fprintf(fid, '- 버퍼 크기: %.1f 초\n', buffer_size/fs);
    fprintf(fid, '- 총 처리 시간: %.1f 초\n', n_buffers/fs);
    fprintf(fid, '- STFT 윈도우: %.1f 초\n\n', stft_window_size/fs);
    
    fprintf(fid, '성능 결과:\n');
    fprintf(fid, '필터뱅크:\n');
    fprintf(fid, '  - 평균: %.2f ms\n', mean(fb_times)*1000);
    fprintf(fid, '  - 최대: %.2f ms\n', max(fb_times)*1000);
    fprintf(fid, '  - 실시간 처리율: %.1f%%\n', sum(fb_times < 1)/length(fb_times)*100);
    
    fprintf(fid, '\nSTFT:\n');
    fprintf(fid, '  - 평균: %.2f ms\n', mean(stft_times)*1000);
    fprintf(fid, '  - 최대: %.2f ms\n', max(stft_times)*1000);
    fprintf(fid, '  - 실시간 처리율: %.1f%%\n', sum(stft_times < 1)/length(stft_times)*100);
    
    fclose(fid);
    fprintf('요약 보고서가 저장되었습니다: %s\n', summary_filename);
end

%% Helper function
function result = iff(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end