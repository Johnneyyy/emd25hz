%% Delay Impact Analysis: Polar H10 vs Device Comparison
% 이 스크립트는 실제 device와 reference인 polar H10 사이에 
% 1-3초 delay가 발생할 경우 성능 측정에 어떤 영향이 발생하는지 분석합니다.
% 
% 분석 내용:
% - ParamALL의 3번째 컬럼 (Polar H10 측정값)을 기준으로 분석
% - 각 컬럼은 1초 후를 의미함
% - 1초, 2초, 3초 후의 HR 값 변화를 분석하는 여러 차트 생성
%
% Author: MATLAB Signal Processing
% Date: 2024

clear all; close all; clc;

%% 1. 데이터 로드 및 확인
fprintf('=== Delay Impact Analysis 시작 ===\n');
fprintf('작업공간에서 ParamALL 변수 확인 중...\n');

% 작업공간에 ParamALL이 있는지 확인
if ~exist('ParamALL', 'var')
    error('작업공간에 ParamALL 변수가 없습니다. 먼저 데이터를 로드해주세요.');
end

% ParamALL 크기 확인
[rows, cols] = size(ParamALL);
fprintf('ParamALL 크기: %d 행 x %d 열\n', rows, cols);

if cols < 3
    error('ParamALL은 최소 3개의 컬럼이 필요합니다.');
end

% 3번째 컬럼 (Polar H10 측정값) 추출
polar_hr = ParamALL(:, 3);
fprintf('Polar H10 HR 데이터 포인트 수: %d\n', length(polar_hr));

% 시간 벡터 생성 (각 포인트는 1초 간격)
time_seconds = (0:length(polar_hr)-1)';

%% 2. Delay별 HR 데이터 준비
fprintf('\nDelay별 HR 데이터 준비 중...\n');

% 1초 delay (4번째 컬럼)
if cols >= 4
    hr_1sec_delay = ParamALL(:, 4);
    fprintf('1초 delay 데이터 사용 가능\n');
else
    warning('4번째 컬럼이 없어 1초 delay 분석을 건너뜁니다.');
    hr_1sec_delay = [];
end

% 2초 delay (5번째 컬럼)
if cols >= 5
    hr_2sec_delay = ParamALL(:, 5);
    fprintf('2초 delay 데이터 사용 가능\n');
else
    warning('5번째 컬럼이 없어 2초 delay 분석을 건너뜁니다.');
    hr_2sec_delay = [];
end

% 3초 delay (6번째 컬럼)
if cols >= 6
    hr_3sec_delay = ParamALL(:, 6);
    fprintf('3초 delay 데이터 사용 가능\n');
else
    warning('6번째 컬럼이 없어 3초 delay 분석을 건너뜁니다.');
    hr_3sec_delay = [];
end

%% 3. 공통 분석 길이 결정
% 모든 데이터의 공통 길이 계산
min_length = length(polar_hr);
if ~isempty(hr_1sec_delay)
    min_length = min(min_length, length(hr_1sec_delay));
end
if ~isempty(hr_2sec_delay)
    min_length = min(min_length, length(hr_2sec_delay));
end
if ~isempty(hr_3sec_delay)
    min_length = min(min_length, length(hr_3sec_delay));
end

fprintf('공통 분석 길이: %d 포인트 (%.1f 초)\n', min_length, min_length);

% 데이터 길이 맞추기
polar_hr = polar_hr(1:min_length);
time_seconds = time_seconds(1:min_length);

if ~isempty(hr_1sec_delay)
    hr_1sec_delay = hr_1sec_delay(1:min_length);
end
if ~isempty(hr_2sec_delay)
    hr_2sec_delay = hr_2sec_delay(1:min_length);
end
if ~isempty(hr_3sec_delay)
    hr_3sec_delay = hr_3sec_delay(1:min_length);
end

%% 4. 통계 분석
fprintf('\n=== 통계 분석 ===\n');

stats = struct();

% Polar H10 기준 통계
stats.polar.mean = mean(polar_hr);
stats.polar.std = std(polar_hr);
stats.polar.min = min(polar_hr);
stats.polar.max = max(polar_hr);
stats.polar.median = median(polar_hr);

fprintf('Polar H10 HR:\n');
fprintf('  평균: %.2f bpm\n', stats.polar.mean);
fprintf('  표준편차: %.2f bpm\n', stats.polar.std);
fprintf('  범위: %.2f - %.2f bpm\n', stats.polar.min, stats.polar.max);
fprintf('  중앙값: %.2f bpm\n', stats.polar.median);

% Delay별 통계 및 오차 계산
if ~isempty(hr_1sec_delay)
    stats.delay1sec.mean = mean(hr_1sec_delay);
    stats.delay1sec.std = std(hr_1sec_delay);
    stats.delay1sec.error = hr_1sec_delay - polar_hr;
    stats.delay1sec.mae = mean(abs(stats.delay1sec.error));
    stats.delay1sec.rmse = sqrt(mean(stats.delay1sec.error.^2));
    stats.delay1sec.correlation = corrcoef(polar_hr, hr_1sec_delay);
    
    fprintf('\n1초 Delay HR:\n');
    fprintf('  평균: %.2f bpm\n', stats.delay1sec.mean);
    fprintf('  표준편차: %.2f bpm\n', stats.delay1sec.std);
    fprintf('  MAE: %.2f bpm\n', stats.delay1sec.mae);
    fprintf('  RMSE: %.2f bpm\n', stats.delay1sec.rmse);
    fprintf('  상관계수: %.4f\n', stats.delay1sec.correlation(1,2));
end

if ~isempty(hr_2sec_delay)
    stats.delay2sec.mean = mean(hr_2sec_delay);
    stats.delay2sec.std = std(hr_2sec_delay);
    stats.delay2sec.error = hr_2sec_delay - polar_hr;
    stats.delay2sec.mae = mean(abs(stats.delay2sec.error));
    stats.delay2sec.rmse = sqrt(mean(stats.delay2sec.error.^2));
    stats.delay2sec.correlation = corrcoef(polar_hr, hr_2sec_delay);
    
    fprintf('\n2초 Delay HR:\n');
    fprintf('  평균: %.2f bpm\n', stats.delay2sec.mean);
    fprintf('  표준편차: %.2f bpm\n', stats.delay2sec.std);
    fprintf('  MAE: %.2f bpm\n', stats.delay2sec.mae);
    fprintf('  RMSE: %.2f bpm\n', stats.delay2sec.rmse);
    fprintf('  상관계수: %.4f\n', stats.delay2sec.correlation(1,2));
end

if ~isempty(hr_3sec_delay)
    stats.delay3sec.mean = mean(hr_3sec_delay);
    stats.delay3sec.std = std(hr_3sec_delay);
    stats.delay3sec.error = hr_3sec_delay - polar_hr;
    stats.delay3sec.mae = mean(abs(stats.delay3sec.error));
    stats.delay3sec.rmse = sqrt(mean(stats.delay3sec.error.^2));
    stats.delay3sec.correlation = corrcoef(polar_hr, hr_3sec_delay);
    
    fprintf('\n3초 Delay HR:\n');
    fprintf('  평균: %.2f bpm\n', stats.delay3sec.mean);
    fprintf('  표준편차: %.2f bpm\n', stats.delay3sec.std);
    fprintf('  MAE: %.2f bpm\n', stats.delay3sec.mae);
    fprintf('  RMSE: %.2f bpm\n', stats.delay3sec.rmse);
    fprintf('  상관계수: %.4f\n', stats.delay3sec.correlation(1,2));
end

%% 5. 1초 Delay 분석 차트 생성
if ~isempty(hr_1sec_delay)
    fprintf('\n1초 Delay 분석 차트 생성 중...\n');
    
    figure('Position', [100, 100, 1600, 1200], 'Name', '1초 Delay 분석');
    
    % 차트 1: 시간에 따른 HR 비교
    subplot(3, 3, 1);
    plot(time_seconds, polar_hr, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Polar H10');
    hold on;
    plot(time_seconds, hr_1sec_delay, 'r--', 'LineWidth', 1.5, 'DisplayName', '1초 Delay');
    xlabel('시간 (초)');
    ylabel('심박수 (bpm)');
    title('시간에 따른 HR 비교: Polar H10 vs 1초 Delay');
    legend('Location', 'best');
    grid on;
    
    % 차트 2: 산점도 (상관관계)
    subplot(3, 3, 2);
    scatter(polar_hr, hr_1sec_delay, 30, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    % 1:1 선 추가
    min_val = min([polar_hr; hr_1sec_delay]);
    max_val = max([polar_hr; hr_1sec_delay]);
    plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.5);
    xlabel('Polar H10 HR (bpm)');
    ylabel('1초 Delay HR (bpm)');
    title(sprintf('상관관계 분석 (R=%.4f)', stats.delay1sec.correlation(1,2)));
    grid on;
    axis equal;
    
    % 차트 3: 오차 분포
    subplot(3, 3, 3);
    histogram(stats.delay1sec.error, 30, 'FaceColor', [0.8, 0.2, 0.2], 'EdgeColor', 'black');
    xlabel('오차 (bpm)');
    ylabel('빈도');
    title(sprintf('오차 분포 (MAE=%.2f, RMSE=%.2f)', stats.delay1sec.mae, stats.delay1sec.rmse));
    grid on;
    hold on;
    xline(0, 'k--', 'LineWidth', 1.5);
    xline(mean(stats.delay1sec.error), 'r--', 'LineWidth', 1.5, 'DisplayName', '평균 오차');
    
    % 차트 4: 시간에 따른 오차
    subplot(3, 3, 4);
    plot(time_seconds, stats.delay1sec.error, 'r-', 'LineWidth', 1);
    hold on;
    yline(0, 'k--', 'LineWidth', 1);
    yline(mean(stats.delay1sec.error), 'r--', 'LineWidth', 1.5, 'DisplayName', '평균');
    xlabel('시간 (초)');
    ylabel('오차 (bpm)');
    title('시간에 따른 오차 변화');
    legend('Location', 'best');
    grid on;
    
    % 차트 5: 누적 오차
    subplot(3, 3, 5);
    cumulative_error = cumsum(abs(stats.delay1sec.error));
    plot(time_seconds, cumulative_error, 'r-', 'LineWidth', 1.5);
    xlabel('시간 (초)');
    ylabel('누적 절대 오차 (bpm)');
    title('누적 절대 오차');
    grid on;
    
    % 차트 6: Bland-Altman Plot
    subplot(3, 3, 6);
    mean_hr = (polar_hr + hr_1sec_delay) / 2;
    diff_hr = hr_1sec_delay - polar_hr;
    scatter(mean_hr, diff_hr, 30, 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    yline(mean(diff_hr), 'r-', 'LineWidth', 1.5, 'DisplayName', sprintf('평균: %.2f', mean(diff_hr)));
    yline(mean(diff_hr) + 1.96*std(diff_hr), 'r--', 'LineWidth', 1.5, 'DisplayName', '+1.96SD');
    yline(mean(diff_hr) - 1.96*std(diff_hr), 'r--', 'LineWidth', 1.5, 'DisplayName', '-1.96SD');
    yline(0, 'k--', 'LineWidth', 1);
    xlabel('평균 HR (bpm)');
    ylabel('차이 (Delay - Polar) (bpm)');
    title('Bland-Altman Plot: 1초 Delay');
    legend('Location', 'best');
    grid on;
    
    % 차트 7: 주파수 도메인 분석 (FFT)
    subplot(3, 3, 7);
    % Polar H10 FFT
    [P_polar, f_polar] = pwelch(polar_hr - mean(polar_hr), [], [], [], 1); % 1Hz 샘플링
    % Delay HR FFT
    [P_delay, f_delay] = pwelch(hr_1sec_delay - mean(hr_1sec_delay), [], [], [], 1);
    semilogy(f_polar, P_polar, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Polar H10');
    hold on;
    semilogy(f_delay, P_delay, 'r--', 'LineWidth', 1.5, 'DisplayName', '1초 Delay');
    xlabel('주파수 (Hz)');
    ylabel('파워 스펙트럼 밀도');
    title('주파수 도메인 분석');
    legend('Location', 'best');
    grid on;
    xlim([0, 0.5]);
    
    % 차트 8: 이동 평균 비교
    subplot(3, 3, 8);
    window_size = min(30, round(length(polar_hr)/10)); % 10% 또는 30포인트
    polar_smooth = movmean(polar_hr, window_size);
    delay_smooth = movmean(hr_1sec_delay, window_size);
    plot(time_seconds, polar_smooth, 'b-', 'LineWidth', 2, 'DisplayName', 'Polar H10 (이동평균)');
    hold on;
    plot(time_seconds, delay_smooth, 'r--', 'LineWidth', 2, 'DisplayName', '1초 Delay (이동평균)');
    xlabel('시간 (초)');
    ylabel('심박수 (bpm)');
    title(sprintf('이동 평균 비교 (윈도우: %d초)', window_size));
    legend('Location', 'best');
    grid on;
    
    % 차트 9: 오차의 절대값 분포
    subplot(3, 3, 9);
    abs_error = abs(stats.delay1sec.error);
    plot(time_seconds, abs_error, 'r-', 'LineWidth', 1);
    hold on;
    yline(mean(abs_error), 'r--', 'LineWidth', 1.5, 'DisplayName', sprintf('평균: %.2f', mean(abs_error)));
    xlabel('시간 (초)');
    ylabel('절대 오차 (bpm)');
    title('시간에 따른 절대 오차');
    legend('Location', 'best');
    grid on;
    
    sgtitle('1초 Delay 영향 분석', 'FontSize', 16, 'FontWeight', 'bold');
end

%% 6. 2초 Delay 분석 차트 생성
if ~isempty(hr_2sec_delay)
    fprintf('2초 Delay 분석 차트 생성 중...\n');
    
    figure('Position', [100, 100, 1600, 1200], 'Name', '2초 Delay 분석');
    
    % 차트 1: 시간에 따른 HR 비교
    subplot(3, 3, 1);
    plot(time_seconds, polar_hr, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Polar H10');
    hold on;
    plot(time_seconds, hr_2sec_delay, 'g--', 'LineWidth', 1.5, 'DisplayName', '2초 Delay');
    xlabel('시간 (초)');
    ylabel('심박수 (bpm)');
    title('시간에 따른 HR 비교: Polar H10 vs 2초 Delay');
    legend('Location', 'best');
    grid on;
    
    % 차트 2: 산점도 (상관관계)
    subplot(3, 3, 2);
    scatter(polar_hr, hr_2sec_delay, 30, 'filled', 'MarkerFaceAlpha', 0.6, 'MarkerFaceColor', [0.2, 0.8, 0.2]);
    hold on;
    min_val = min([polar_hr; hr_2sec_delay]);
    max_val = max([polar_hr; hr_2sec_delay]);
    plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.5);
    xlabel('Polar H10 HR (bpm)');
    ylabel('2초 Delay HR (bpm)');
    title(sprintf('상관관계 분석 (R=%.4f)', stats.delay2sec.correlation(1,2)));
    grid on;
    axis equal;
    
    % 차트 3: 오차 분포
    subplot(3, 3, 3);
    histogram(stats.delay2sec.error, 30, 'FaceColor', [0.2, 0.8, 0.2], 'EdgeColor', 'black');
    xlabel('오차 (bpm)');
    ylabel('빈도');
    title(sprintf('오차 분포 (MAE=%.2f, RMSE=%.2f)', stats.delay2sec.mae, stats.delay2sec.rmse));
    grid on;
    hold on;
    xline(0, 'k--', 'LineWidth', 1.5);
    xline(mean(stats.delay2sec.error), 'g--', 'LineWidth', 1.5, 'DisplayName', '평균 오차');
    
    % 차트 4: 시간에 따른 오차
    subplot(3, 3, 4);
    plot(time_seconds, stats.delay2sec.error, 'g-', 'LineWidth', 1);
    hold on;
    yline(0, 'k--', 'LineWidth', 1);
    yline(mean(stats.delay2sec.error), 'g--', 'LineWidth', 1.5, 'DisplayName', '평균');
    xlabel('시간 (초)');
    ylabel('오차 (bpm)');
    title('시간에 따른 오차 변화');
    legend('Location', 'best');
    grid on;
    
    % 차트 5: 누적 오차
    subplot(3, 3, 5);
    cumulative_error = cumsum(abs(stats.delay2sec.error));
    plot(time_seconds, cumulative_error, 'g-', 'LineWidth', 1.5);
    xlabel('시간 (초)');
    ylabel('누적 절대 오차 (bpm)');
    title('누적 절대 오차');
    grid on;
    
    % 차트 6: Bland-Altman Plot
    subplot(3, 3, 6);
    mean_hr = (polar_hr + hr_2sec_delay) / 2;
    diff_hr = hr_2sec_delay - polar_hr;
    scatter(mean_hr, diff_hr, 30, 'filled', 'MarkerFaceAlpha', 0.6, 'MarkerFaceColor', [0.2, 0.8, 0.2]);
    hold on;
    yline(mean(diff_hr), 'g-', 'LineWidth', 1.5, 'DisplayName', sprintf('평균: %.2f', mean(diff_hr)));
    yline(mean(diff_hr) + 1.96*std(diff_hr), 'g--', 'LineWidth', 1.5, 'DisplayName', '+1.96SD');
    yline(mean(diff_hr) - 1.96*std(diff_hr), 'g--', 'LineWidth', 1.5, 'DisplayName', '-1.96SD');
    yline(0, 'k--', 'LineWidth', 1);
    xlabel('평균 HR (bpm)');
    ylabel('차이 (Delay - Polar) (bpm)');
    title('Bland-Altman Plot: 2초 Delay');
    legend('Location', 'best');
    grid on;
    
    % 차트 7: 주파수 도메인 분석 (FFT)
    subplot(3, 3, 7);
    [P_polar, f_polar] = pwelch(polar_hr - mean(polar_hr), [], [], [], 1);
    [P_delay, f_delay] = pwelch(hr_2sec_delay - mean(hr_2sec_delay), [], [], [], 1);
    semilogy(f_polar, P_polar, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Polar H10');
    hold on;
    semilogy(f_delay, P_delay, 'g--', 'LineWidth', 1.5, 'DisplayName', '2초 Delay');
    xlabel('주파수 (Hz)');
    ylabel('파워 스펙트럼 밀도');
    title('주파수 도메인 분석');
    legend('Location', 'best');
    grid on;
    xlim([0, 0.5]);
    
    % 차트 8: 이동 평균 비교
    subplot(3, 3, 8);
    window_size = min(30, round(length(polar_hr)/10));
    polar_smooth = movmean(polar_hr, window_size);
    delay_smooth = movmean(hr_2sec_delay, window_size);
    plot(time_seconds, polar_smooth, 'b-', 'LineWidth', 2, 'DisplayName', 'Polar H10 (이동평균)');
    hold on;
    plot(time_seconds, delay_smooth, 'g--', 'LineWidth', 2, 'DisplayName', '2초 Delay (이동평균)');
    xlabel('시간 (초)');
    ylabel('심박수 (bpm)');
    title(sprintf('이동 평균 비교 (윈도우: %d초)', window_size));
    legend('Location', 'best');
    grid on;
    
    % 차트 9: 오차의 절대값 분포
    subplot(3, 3, 9);
    abs_error = abs(stats.delay2sec.error);
    plot(time_seconds, abs_error, 'g-', 'LineWidth', 1);
    hold on;
    yline(mean(abs_error), 'g--', 'LineWidth', 1.5, 'DisplayName', sprintf('평균: %.2f', mean(abs_error)));
    xlabel('시간 (초)');
    ylabel('절대 오차 (bpm)');
    title('시간에 따른 절대 오차');
    legend('Location', 'best');
    grid on;
    
    sgtitle('2초 Delay 영향 분석', 'FontSize', 16, 'FontWeight', 'bold');
end

%% 7. 3초 Delay 분석 차트 생성
if ~isempty(hr_3sec_delay)
    fprintf('3초 Delay 분석 차트 생성 중...\n');
    
    figure('Position', [100, 100, 1600, 1200], 'Name', '3초 Delay 분석');
    
    % 차트 1: 시간에 따른 HR 비교
    subplot(3, 3, 1);
    plot(time_seconds, polar_hr, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Polar H10');
    hold on;
    plot(time_seconds, hr_3sec_delay, 'm--', 'LineWidth', 1.5, 'DisplayName', '3초 Delay');
    xlabel('시간 (초)');
    ylabel('심박수 (bpm)');
    title('시간에 따른 HR 비교: Polar H10 vs 3초 Delay');
    legend('Location', 'best');
    grid on;
    
    % 차트 2: 산점도 (상관관계)
    subplot(3, 3, 2);
    scatter(polar_hr, hr_3sec_delay, 30, 'filled', 'MarkerFaceAlpha', 0.6, 'MarkerFaceColor', [0.8, 0.2, 0.8]);
    hold on;
    min_val = min([polar_hr; hr_3sec_delay]);
    max_val = max([polar_hr; hr_3sec_delay]);
    plot([min_val, max_val], [min_val, max_val], 'k--', 'LineWidth', 1.5);
    xlabel('Polar H10 HR (bpm)');
    ylabel('3초 Delay HR (bpm)');
    title(sprintf('상관관계 분석 (R=%.4f)', stats.delay3sec.correlation(1,2)));
    grid on;
    axis equal;
    
    % 차트 3: 오차 분포
    subplot(3, 3, 3);
    histogram(stats.delay3sec.error, 30, 'FaceColor', [0.8, 0.2, 0.8], 'EdgeColor', 'black');
    xlabel('오차 (bpm)');
    ylabel('빈도');
    title(sprintf('오차 분포 (MAE=%.2f, RMSE=%.2f)', stats.delay3sec.mae, stats.delay3sec.rmse));
    grid on;
    hold on;
    xline(0, 'k--', 'LineWidth', 1.5);
    xline(mean(stats.delay3sec.error), 'm--', 'LineWidth', 1.5, 'DisplayName', '평균 오차');
    
    % 차트 4: 시간에 따른 오차
    subplot(3, 3, 4);
    plot(time_seconds, stats.delay3sec.error, 'm-', 'LineWidth', 1);
    hold on;
    yline(0, 'k--', 'LineWidth', 1);
    yline(mean(stats.delay3sec.error), 'm--', 'LineWidth', 1.5, 'DisplayName', '평균');
    xlabel('시간 (초)');
    ylabel('오차 (bpm)');
    title('시간에 따른 오차 변화');
    legend('Location', 'best');
    grid on;
    
    % 차트 5: 누적 오차
    subplot(3, 3, 5);
    cumulative_error = cumsum(abs(stats.delay3sec.error));
    plot(time_seconds, cumulative_error, 'm-', 'LineWidth', 1.5);
    xlabel('시간 (초)');
    ylabel('누적 절대 오차 (bpm)');
    title('누적 절대 오차');
    grid on;
    
    % 차트 6: Bland-Altman Plot
    subplot(3, 3, 6);
    mean_hr = (polar_hr + hr_3sec_delay) / 2;
    diff_hr = hr_3sec_delay - polar_hr;
    scatter(mean_hr, diff_hr, 30, 'filled', 'MarkerFaceAlpha', 0.6, 'MarkerFaceColor', [0.8, 0.2, 0.8]);
    hold on;
    yline(mean(diff_hr), 'm-', 'LineWidth', 1.5, 'DisplayName', sprintf('평균: %.2f', mean(diff_hr)));
    yline(mean(diff_hr) + 1.96*std(diff_hr), 'm--', 'LineWidth', 1.5, 'DisplayName', '+1.96SD');
    yline(mean(diff_hr) - 1.96*std(diff_hr), 'm--', 'LineWidth', 1.5, 'DisplayName', '-1.96SD');
    yline(0, 'k--', 'LineWidth', 1);
    xlabel('평균 HR (bpm)');
    ylabel('차이 (Delay - Polar) (bpm)');
    title('Bland-Altman Plot: 3초 Delay');
    legend('Location', 'best');
    grid on;
    
    % 차트 7: 주파수 도메인 분석 (FFT)
    subplot(3, 3, 7);
    [P_polar, f_polar] = pwelch(polar_hr - mean(polar_hr), [], [], [], 1);
    [P_delay, f_delay] = pwelch(hr_3sec_delay - mean(hr_3sec_delay), [], [], [], 1);
    semilogy(f_polar, P_polar, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Polar H10');
    hold on;
    semilogy(f_delay, P_delay, 'm--', 'LineWidth', 1.5, 'DisplayName', '3초 Delay');
    xlabel('주파수 (Hz)');
    ylabel('파워 스펙트럼 밀도');
    title('주파수 도메인 분석');
    legend('Location', 'best');
    grid on;
    xlim([0, 0.5]);
    
    % 차트 8: 이동 평균 비교
    subplot(3, 3, 8);
    window_size = min(30, round(length(polar_hr)/10));
    polar_smooth = movmean(polar_hr, window_size);
    delay_smooth = movmean(hr_3sec_delay, window_size);
    plot(time_seconds, polar_smooth, 'b-', 'LineWidth', 2, 'DisplayName', 'Polar H10 (이동평균)');
    hold on;
    plot(time_seconds, delay_smooth, 'm--', 'LineWidth', 2, 'DisplayName', '3초 Delay (이동평균)');
    xlabel('시간 (초)');
    ylabel('심박수 (bpm)');
    title(sprintf('이동 평균 비교 (윈도우: %d초)', window_size));
    legend('Location', 'best');
    grid on;
    
    % 차트 9: 오차의 절대값 분포
    subplot(3, 3, 9);
    abs_error = abs(stats.delay3sec.error);
    plot(time_seconds, abs_error, 'm-', 'LineWidth', 1);
    hold on;
    yline(mean(abs_error), 'm--', 'LineWidth', 1.5, 'DisplayName', sprintf('평균: %.2f', mean(abs_error)));
    xlabel('시간 (초)');
    ylabel('절대 오차 (bpm)');
    title('시간에 따른 절대 오차');
    legend('Location', 'best');
    grid on;
    
    sgtitle('3초 Delay 영향 분석', 'FontSize', 16, 'FontWeight', 'bold');
end

%% 8. 종합 비교 차트
fprintf('\n종합 비교 차트 생성 중...\n');

figure('Position', [100, 100, 1800, 1000], 'Name', 'Delay 종합 비교');

% 차트 1: 모든 Delay를 한 번에 비교
subplot(2, 3, 1);
plot(time_seconds, polar_hr, 'b-', 'LineWidth', 2, 'DisplayName', 'Polar H10 (Reference)');
hold on;
if ~isempty(hr_1sec_delay)
    plot(time_seconds, hr_1sec_delay, 'r--', 'LineWidth', 1.5, 'DisplayName', '1초 Delay');
end
if ~isempty(hr_2sec_delay)
    plot(time_seconds, hr_2sec_delay, 'g--', 'LineWidth', 1.5, 'DisplayName', '2초 Delay');
end
if ~isempty(hr_3sec_delay)
    plot(time_seconds, hr_3sec_delay, 'm--', 'LineWidth', 1.5, 'DisplayName', '3초 Delay');
end
xlabel('시간 (초)');
ylabel('심박수 (bpm)');
title('모든 Delay 비교');
legend('Location', 'best');
grid on;

% 차트 2: 오차 비교
subplot(2, 3, 2);
hold on;
if ~isempty(hr_1sec_delay)
    plot(time_seconds, abs(stats.delay1sec.error), 'r-', 'LineWidth', 1.5, 'DisplayName', '1초 Delay');
end
if ~isempty(hr_2sec_delay)
    plot(time_seconds, abs(stats.delay2sec.error), 'g-', 'LineWidth', 1.5, 'DisplayName', '2초 Delay');
end
if ~isempty(hr_3sec_delay)
    plot(time_seconds, abs(stats.delay3sec.error), 'm-', 'LineWidth', 1.5, 'DisplayName', '3초 Delay');
end
xlabel('시간 (초)');
ylabel('절대 오차 (bpm)');
title('절대 오차 비교');
legend('Location', 'best');
grid on;

% 차트 3: 성능 지표 비교 (Bar Chart)
subplot(2, 3, 3);
delay_labels = {};
mae_values = [];
rmse_values = [];
corr_values = [];

if ~isempty(hr_1sec_delay)
    delay_labels{end+1} = '1초';
    mae_values(end+1) = stats.delay1sec.mae;
    rmse_values(end+1) = stats.delay1sec.rmse;
    corr_values(end+1) = stats.delay1sec.correlation(1,2);
end
if ~isempty(hr_2sec_delay)
    delay_labels{end+1} = '2초';
    mae_values(end+1) = stats.delay2sec.mae;
    rmse_values(end+1) = stats.delay2sec.rmse;
    corr_values(end+1) = stats.delay2sec.correlation(1,2);
end
if ~isempty(hr_3sec_delay)
    delay_labels{end+1} = '3초';
    mae_values(end+1) = stats.delay3sec.mae;
    rmse_values(end+1) = stats.delay3sec.rmse;
    corr_values(end+1) = stats.delay3sec.correlation(1,2);
end

x = 1:length(delay_labels);
width = 0.25;
bar(x - width, mae_values, width, 'FaceColor', [0.8, 0.2, 0.2], 'DisplayName', 'MAE');
hold on;
bar(x, rmse_values, width, 'FaceColor', [0.2, 0.6, 0.8], 'DisplayName', 'RMSE');
set(gca, 'XTick', x, 'XTickLabel', delay_labels);
xlabel('Delay 시간');
ylabel('오차 (bpm)');
title('성능 지표 비교');
legend('Location', 'best');
grid on;

% 차트 4: 상관계수 비교
subplot(2, 3, 4);
bar(x, corr_values, 'FaceColor', [0.2, 0.8, 0.2]);
set(gca, 'XTick', x, 'XTickLabel', delay_labels);
xlabel('Delay 시간');
ylabel('상관계수');
title('상관계수 비교');
ylim([0, 1]);
grid on;
hold on;
for i = 1:length(corr_values)
    text(i, corr_values(i) + 0.02, sprintf('%.4f', corr_values(i)), ...
         'HorizontalAlignment', 'center', 'FontSize', 10);
end

% 차트 5: 누적 오차 비교
subplot(2, 3, 5);
hold on;
if ~isempty(hr_1sec_delay)
    plot(time_seconds, cumsum(abs(stats.delay1sec.error)), 'r-', 'LineWidth', 1.5, 'DisplayName', '1초 Delay');
end
if ~isempty(hr_2sec_delay)
    plot(time_seconds, cumsum(abs(stats.delay2sec.error)), 'g-', 'LineWidth', 1.5, 'DisplayName', '2초 Delay');
end
if ~isempty(hr_3sec_delay)
    plot(time_seconds, cumsum(abs(stats.delay3sec.error)), 'm-', 'LineWidth', 1.5, 'DisplayName', '3초 Delay');
end
xlabel('시간 (초)');
ylabel('누적 절대 오차 (bpm)');
title('누적 절대 오차 비교');
legend('Location', 'best');
grid on;

% 차트 6: 오차 분포 비교 (Box Plot)
subplot(2, 3, 6);
error_data = [];
group_data = [];
if ~isempty(hr_1sec_delay)
    error_data = [error_data; abs(stats.delay1sec.error)];
    group_data = [group_data; ones(length(stats.delay1sec.error), 1)];
end
if ~isempty(hr_2sec_delay)
    error_data = [error_data; abs(stats.delay2sec.error)];
    group_data = [group_data; 2*ones(length(stats.delay2sec.error), 1)];
end
if ~isempty(hr_3sec_delay)
    error_data = [error_data; abs(stats.delay3sec.error)];
    group_data = [group_data; 3*ones(length(stats.delay3sec.error), 1)];
end
boxplot(error_data, group_data, 'Labels', delay_labels);
ylabel('절대 오차 (bpm)');
title('오차 분포 비교 (Box Plot)');
grid on;

sgtitle('Delay 영향 종합 분석', 'FontSize', 16, 'FontWeight', 'bold');

%% 9. 결과 저장
fprintf('\n결과 저장 중...\n');

% 분석 결과 구조체 생성
results.polar_hr = polar_hr;
results.time_seconds = time_seconds;
results.stats = stats;

if ~isempty(hr_1sec_delay)
    results.delay1sec.hr = hr_1sec_delay;
    results.delay1sec.error = stats.delay1sec.error;
end
if ~isempty(hr_2sec_delay)
    results.delay2sec.hr = hr_2sec_delay;
    results.delay2sec.error = stats.delay2sec.error;
end
if ~isempty(hr_3sec_delay)
    results.delay3sec.hr = hr_3sec_delay;
    results.delay3sec.error = stats.delay3sec.error;
end

% MAT 파일로 저장
save('delay_impact_analysis_results.mat', 'results');
fprintf('결과가 delay_impact_analysis_results.mat 파일로 저장되었습니다.\n');

% CSV 파일로도 저장
if ~isempty(hr_1sec_delay) || ~isempty(hr_2sec_delay) || ~isempty(hr_3sec_delay)
    csv_data = table(time_seconds, polar_hr, 'VariableNames', {'Time_sec', 'Polar_H10_HR'});
    if ~isempty(hr_1sec_delay)
        csv_data.Delay_1sec_HR = hr_1sec_delay;
        csv_data.Delay_1sec_Error = stats.delay1sec.error;
    end
    if ~isempty(hr_2sec_delay)
        csv_data.Delay_2sec_HR = hr_2sec_delay;
        csv_data.Delay_2sec_Error = stats.delay2sec.error;
    end
    if ~isempty(hr_3sec_delay)
        csv_data.Delay_3sec_HR = hr_3sec_delay;
        csv_data.Delay_3sec_Error = stats.delay3sec.error;
    end
    writetable(csv_data, 'delay_impact_analysis_results.csv');
    fprintf('결과가 delay_impact_analysis_results.csv 파일로 저장되었습니다.\n');
end

%% 10. 최종 요약 출력
fprintf('\n========== 분석 완료 ==========\n');
fprintf('생성된 차트:\n');
if ~isempty(hr_1sec_delay)
    fprintf('  - 1초 Delay 분석 차트 (9개)\n');
end
if ~isempty(hr_2sec_delay)
    fprintf('  - 2초 Delay 분석 차트 (9개)\n');
end
if ~isempty(hr_3sec_delay)
    fprintf('  - 3초 Delay 분석 차트 (9개)\n');
end
fprintf('  - 종합 비교 차트 (6개)\n');
fprintf('\n총 %d개의 차트가 생성되었습니다.\n', ...
    (~isempty(hr_1sec_delay))*9 + (~isempty(hr_2sec_delay))*9 + (~isempty(hr_3sec_delay))*9 + 6);

fprintf('\n성능 요약:\n');
fprintf('%-10s | %-8s | %-8s | %-10s\n', 'Delay', 'MAE', 'RMSE', '상관계수');
fprintf('--------------------------------------------------------\n');
if ~isempty(hr_1sec_delay)
    fprintf('%-10s | %8.2f | %8.2f | %10.4f\n', '1초', ...
        stats.delay1sec.mae, stats.delay1sec.rmse, stats.delay1sec.correlation(1,2));
end
if ~isempty(hr_2sec_delay)
    fprintf('%-10s | %8.2f | %8.2f | %10.4f\n', '2초', ...
        stats.delay2sec.mae, stats.delay2sec.rmse, stats.delay2sec.correlation(1,2));
end
if ~isempty(hr_3sec_delay)
    fprintf('%-10s | %8.2f | %8.2f | %10.4f\n', '3초', ...
        stats.delay3sec.mae, stats.delay3sec.rmse, stats.delay3sec.correlation(1,2));
end

fprintf('\n분석이 완료되었습니다!\n');
