%% 스마트워치 가속도계 데이터로부터 이동 속도 계산 테스트 스크립트
% 이 스크립트는 calculate_movement_speed 함수를 테스트합니다.
%
% 작성일: 2024

clear all; close all; clc;

%% 1. 샘플 데이터 생성
fprintf('=== 샘플 가속도계 데이터 생성 ===\n');
fs = 25; % 25 Hz 샘플링 주파수
duration = 30; % 30초 데이터

% 다양한 활동 시나리오 테스트
scenarios = {'walking', 'running', 'mixed', 'watch_viewing'};
scenario_names = {'걷기', '달리기', '혼합 (걷기+달리기)', '워치를 보면서 뛰기'};

for scenario_idx = 1:length(scenarios)
    fprintf('\n--- 시나리오 %d: %s ---\n', scenario_idx, scenario_names{scenario_idx});
    
    % 샘플 데이터 생성
    [acc_x, acc_y, acc_z, true_speed] = generate_sample_accelerometer_data(...
        fs, duration, scenarios{scenario_idx}, ...
        'walking_speed', 1.4, 'running_speed', 3.0, 'noise_level', 0.2);
    
    %% 2. 속도 계산
    fprintf('속도 계산 중...\n');
    [speed, velocity, filtered_acc, results] = calculate_movement_speed(...
        acc_x, acc_y, acc_z, fs, ...
        'gravity_removal', 'highpass', ...
        'cutoff_freq', 0.1, ...
        'drift_correction', 'highpass', ...
        'integration_window', 5, ...
        'min_speed_threshold', 0.1);
    
    %% 3. 결과 시각화
    figure('Position', [100 + scenario_idx*50, 100 + scenario_idx*50, 1400, 900], ...
           'Name', sprintf('속도 계산 결과: %s', scenario_names{scenario_idx}));
    
    t = results.time;
    
    % 서브플롯 1: 원본 가속도 데이터
    subplot(4, 2, 1);
    plot(t, acc_x, 'r-', 'LineWidth', 0.5); hold on;
    plot(t, acc_y, 'g-', 'LineWidth', 0.5);
    plot(t, acc_z, 'b-', 'LineWidth', 0.5);
    xlabel('시간 (초)');
    ylabel('가속도 (m/s²)');
    title('원본 가속도계 데이터');
    legend('X축', 'Y축', 'Z축', 'Location', 'best');
    grid on;
    
    % 서브플롯 2: 필터링된 가속도 데이터
    subplot(4, 2, 2);
    plot(t, filtered_acc(:, 1), 'r-', 'LineWidth', 1); hold on;
    plot(t, filtered_acc(:, 2), 'g-', 'LineWidth', 1);
    plot(t, filtered_acc(:, 3), 'b-', 'LineWidth', 1);
    xlabel('시간 (초)');
    ylabel('가속도 (m/s²)');
    title('필터링된 가속도 데이터');
    legend('X축', 'Y축', 'Z축', 'Location', 'best');
    grid on;
    
    % 서브플롯 3: 가속도 크기
    subplot(4, 2, 3);
    plot(t, results.acceleration.magnitude, 'k-', 'LineWidth', 1); hold on;
    plot(t, results.acceleration.dynamic, 'r-', 'LineWidth', 1);
    xlabel('시간 (초)');
    ylabel('가속도 (m/s²)');
    title('가속도 벡터 크기');
    legend('전체 크기', '동적 성분', 'Location', 'best');
    grid on;
    
    % 서브플롯 4: 활동 분류
    subplot(4, 2, 4);
    colors = [0.8 0.8 0.8; 0.2 0.6 0.8; 0.8 0.2 0.2]; % 회색: 정지, 파란색: 걷기, 빨간색: 달리기
    activity_colors = colors(results.activity_type + 1, :);
    scatter(t, results.movement_energy, 20, activity_colors, 'filled');
    xlabel('시간 (초)');
    ylabel('움직임 에너지');
    title('활동 분류 (회색: 정지, 파란색: 걷기, 빨간색: 달리기)');
    grid on;
    
    % 서브플롯 5: 속도 벡터 (X, Y, Z)
    subplot(4, 2, 5);
    plot(t, velocity(:, 1), 'r-', 'LineWidth', 1); hold on;
    plot(t, velocity(:, 2), 'g-', 'LineWidth', 1);
    plot(t, velocity(:, 3), 'b-', 'LineWidth', 1);
    xlabel('시간 (초)');
    ylabel('속도 (m/s)');
    title('속도 벡터 (X, Y, Z)');
    legend('V_x', 'V_y', 'V_z', 'Location', 'best');
    grid on;
    
    % 서브플롯 6: 계산된 속도 vs 실제 속도
    subplot(4, 2, 6);
    plot(t, speed, 'b-', 'LineWidth', 2); hold on;
    plot(t, true_speed, 'r--', 'LineWidth', 2);
    xlabel('시간 (초)');
    ylabel('속도 (m/s)');
    title('계산된 속도 vs 실제 속도');
    legend('계산된 속도', '실제 속도', 'Location', 'best');
    grid on;
    
    % 서브플롯 7: 속도 비교 (km/h)
    subplot(4, 2, 7);
    speed_kmh = speed * 3.6;
    true_speed_kmh = true_speed * 3.6;
    plot(t, speed_kmh, 'b-', 'LineWidth', 2); hold on;
    plot(t, true_speed_kmh, 'r--', 'LineWidth', 2);
    xlabel('시간 (초)');
    ylabel('속도 (km/h)');
    title('속도 비교 (km/h)');
    legend('계산된 속도', '실제 속도', 'Location', 'best');
    grid on;
    
    % 서브플롯 8: 오차 분석
    subplot(4, 2, 8);
    error = speed - true_speed;
    error_percent = 100 * error ./ (true_speed + eps); % 상대 오차 (%)
    plot(t, error_percent, 'k-', 'LineWidth', 1);
    xlabel('시간 (초)');
    ylabel('상대 오차 (%)');
    title('속도 계산 오차');
    grid on;
    yline(0, 'r--', 'LineWidth', 1);
    
    %% 4. 통계 출력
    fprintf('\n=== 결과 통계 ===\n');
    fprintf('실제 평균 속도: %.2f m/s (%.2f km/h)\n', mean(true_speed), mean(true_speed)*3.6);
    fprintf('계산 평균 속도: %.2f m/s (%.2f km/h)\n', mean(speed), mean(speed)*3.6);
    fprintf('실제 최대 속도: %.2f m/s (%.2f km/h)\n', max(true_speed), max(true_speed)*3.6);
    fprintf('계산 최대 속도: %.2f m/s (%.2f km/h)\n', max(speed), max(speed)*3.6);
    
    % RMSE 계산
    rmse = sqrt(mean((speed - true_speed).^2));
    fprintf('RMSE: %.3f m/s\n', rmse);
    
    % 상관계수
    valid_idx = ~isnan(speed) & ~isnan(true_speed) & (true_speed > 0);
    if sum(valid_idx) > 10
        correlation = corrcoef(speed(valid_idx), true_speed(valid_idx));
        fprintf('상관계수: %.3f\n', correlation(1,2));
    end
    
    fprintf('정지 비율: %.1f%%, 걷기 비율: %.1f%%, 달리기 비율: %.1f%%\n', ...
        results.statistics.stationary_ratio * 100, ...
        results.statistics.walking_ratio * 100, ...
        results.statistics.running_ratio * 100);
    
    %% 5. 결과 저장
    save_filename = sprintf('speed_calculation_results_%s.mat', scenarios{scenario_idx});
    save(save_filename, 'acc_x', 'acc_y', 'acc_z', 'speed', 'velocity', ...
         'filtered_acc', 'results', 'true_speed', 't', 'fs');
    fprintf('결과가 "%s"에 저장되었습니다.\n', save_filename);
end

fprintf('\n=== 모든 시나리오 테스트 완료 ===\n');
