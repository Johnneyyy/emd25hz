%% 실제 CSV 파일에서 가속도계 데이터를 읽어 속도 계산하는 예제
% 이 스크립트는 CSV 파일에서 가속도계 데이터를 읽어 속도를 계산합니다.
%
% CSV 파일 형식:
%   - 첫 번째 열: 시간 (초) 또는 샘플 번호
%   - 두 번째 열: acc_x (가속도 x축)
%   - 세 번째 열: acc_y (가속도 y축)
%   - 네 번째 열: acc_z (가속도 z축)
%
% 또는:
%   - 헤더가 있는 경우: 'Time', 'AccX', 'AccY', 'AccZ' 등의 컬럼명 사용

clear all; close all; clc;

%% 1. CSV 파일에서 데이터 읽기
csv_filename = 'accelerometer_data.csv'; % 실제 파일명으로 변경하세요

fprintf('CSV 파일에서 데이터 읽기: %s\n', csv_filename);

try
    % 테이블 형식으로 읽기 (헤더가 있는 경우)
    data_table = readtable(csv_filename);
    
    % 컬럼명 확인 및 데이터 추출
    var_names = data_table.Properties.VariableNames;
    
    % 일반적인 컬럼명 패턴 매칭
    acc_x_col = [];
    acc_y_col = [];
    acc_z_col = [];
    
    for i = 1:length(var_names)
        col_name = lower(var_names{i});
        if contains(col_name, 'x') && (contains(col_name, 'acc') || contains(col_name, 'ax'))
            acc_x_col = i;
        elseif contains(col_name, 'y') && (contains(col_name, 'acc') || contains(col_name, 'ay'))
            acc_y_col = i;
        elseif contains(col_name, 'z') && (contains(col_name, 'acc') || contains(col_name, 'az'))
            acc_z_col = i;
        end
    end
    
    if isempty(acc_x_col) || isempty(acc_y_col) || isempty(acc_z_col)
        % 컬럼명을 찾지 못한 경우, 순서대로 가정 (2, 3, 4번째 컬럼)
        fprintf('컬럼명을 찾지 못했습니다. 순서대로 읽습니다.\n');
        acc_x = table2array(data_table(:, 2));
        acc_y = table2array(data_table(:, 3));
        acc_z = table2array(data_table(:, 4));
    else
        acc_x = table2array(data_table(:, acc_x_col));
        acc_y = table2array(data_table(:, acc_y_col));
        acc_z = table2array(data_table(:, acc_z_col));
    end
    
catch
    % 테이블 읽기 실패 시, 숫자 배열로 읽기
    fprintf('테이블 형식 읽기 실패. 숫자 배열로 읽습니다.\n');
    data = readmatrix(csv_filename);
    
    % 데이터 형식 가정: [시간, acc_x, acc_y, acc_z] 또는 [acc_x, acc_y, acc_z]
    if size(data, 2) >= 4
        acc_x = data(:, 2);
        acc_y = data(:, 3);
        acc_z = data(:, 4);
    elseif size(data, 2) == 3
        acc_x = data(:, 1);
        acc_y = data(:, 2);
        acc_z = data(:, 3);
    else
        error('CSV 파일 형식이 올바르지 않습니다. 최소 3개의 컬럼(acc_x, acc_y, acc_z)이 필요합니다.');
    end
end

% NaN 값 제거
valid_idx = ~isnan(acc_x) & ~isnan(acc_y) & ~isnan(acc_z);
acc_x = acc_x(valid_idx);
acc_y = acc_y(valid_idx);
acc_z = acc_z(valid_idx);

fprintf('데이터 읽기 완료: %d 샘플\n', length(acc_x));

%% 2. 샘플링 주파수 설정
% CSV 파일에 시간 정보가 있는 경우
if exist('data_table', 'var') && any(contains(lower(var_names), 'time'))
    time_col = find(contains(lower(var_names), 'time'), 1);
    time_data = table2array(data_table(:, time_col));
    time_data = time_data(valid_idx);
    
    % 샘플링 주파수 계산
    if length(time_data) > 1
        dt = mean(diff(time_data));
        fs = 1 / dt;
        fprintf('계산된 샘플링 주파수: %.2f Hz\n', fs);
    else
        fs = 25; % 기본값
        fprintf('시간 정보가 부족합니다. 기본 샘플링 주파수 사용: %.1f Hz\n', fs);
    end
else
    % 시간 정보가 없는 경우, 기본값 사용
    fs = 25; % 25 Hz (스마트워치 일반 샘플링 주파수)
    fprintf('시간 정보가 없습니다. 기본 샘플링 주파수 사용: %.1f Hz\n', fs);
end

%% 3. 속도 계산
fprintf('\n속도 계산 시작...\n');

[speed, velocity, filtered_acc, results] = calculate_movement_speed(...
    acc_x, acc_y, acc_z, fs, ...
    'gravity_removal', 'highpass', ...
    'cutoff_freq', 0.1, ...
    'drift_correction', 'highpass', ...
    'integration_window', 5, ...
    'min_speed_threshold', 0.1);

%% 4. 결과 시각화
figure('Position', [100, 100, 1400, 800], 'Name', '가속도계 데이터 속도 계산 결과');

t = results.time;

% 서브플롯 1: 원본 가속도 데이터
subplot(3, 2, 1);
plot(t, acc_x, 'r-', 'LineWidth', 0.5); hold on;
plot(t, acc_y, 'g-', 'LineWidth', 0.5);
plot(t, acc_z, 'b-', 'LineWidth', 0.5);
xlabel('시간 (초)');
ylabel('가속도 (m/s²)');
title('원본 가속도계 데이터');
legend('X축', 'Y축', 'Z축', 'Location', 'best');
grid on;

% 서브플롯 2: 필터링된 가속도 데이터
subplot(3, 2, 2);
plot(t, filtered_acc(:, 1), 'r-', 'LineWidth', 1); hold on;
plot(t, filtered_acc(:, 2), 'g-', 'LineWidth', 1);
plot(t, filtered_acc(:, 3), 'b-', 'LineWidth', 1);
xlabel('시간 (초)');
ylabel('가속도 (m/s²)');
title('필터링된 가속도 데이터');
legend('X축', 'Y축', 'Z축', 'Location', 'best');
grid on;

% 서브플롯 3: 가속도 크기
subplot(3, 2, 3);
plot(t, results.acceleration.magnitude, 'k-', 'LineWidth', 1); hold on;
plot(t, results.acceleration.dynamic, 'r-', 'LineWidth', 1);
xlabel('시간 (초)');
ylabel('가속도 (m/s²)');
title('가속도 벡터 크기');
legend('전체 크기', '동적 성분', 'Location', 'best');
grid on;

% 서브플롯 4: 활동 분류
subplot(3, 2, 4);
colors = [0.8 0.8 0.8; 0.2 0.6 0.8; 0.8 0.2 0.2]; % 회색: 정지, 파란색: 걷기, 빨간색: 달리기
activity_colors = colors(results.activity_type + 1, :);
scatter(t, results.movement_energy, 20, activity_colors, 'filled');
xlabel('시간 (초)');
ylabel('움직임 에너지');
title('활동 분류 (회색: 정지, 파란색: 걷기, 빨간색: 달리기)');
grid on;

% 서브플롯 5: 계산된 속도
subplot(3, 2, 5);
plot(t, speed, 'b-', 'LineWidth', 2);
xlabel('시간 (초)');
ylabel('속도 (m/s)');
title('계산된 이동 속도');
grid on;

% 서브플롯 6: 속도 (km/h)
subplot(3, 2, 6);
speed_kmh = speed * 3.6;
plot(t, speed_kmh, 'b-', 'LineWidth', 2);
xlabel('시간 (초)');
ylabel('속도 (km/h)');
title('계산된 이동 속도 (km/h)');
grid on;

%% 5. 결과 저장
fprintf('\n결과 저장 중...\n');

% 결과를 CSV 파일로 저장
output_table = table(t', speed', speed_kmh', ...
    'VariableNames', {'Time_s', 'Speed_mps', 'Speed_kmh'});
writetable(output_table, 'calculated_speed_results.csv');
fprintf('속도 결과가 "calculated_speed_results.csv"에 저장되었습니다.\n');

% 전체 결과를 MAT 파일로 저장
save('speed_calculation_results.mat', 'speed', 'velocity', 'filtered_acc', ...
     'results', 'acc_x', 'acc_y', 'acc_z', 't', 'fs');
fprintf('전체 결과가 "speed_calculation_results.mat"에 저장되었습니다.\n');

%% 6. 통계 정보 출력
fprintf('\n=== 결과 통계 ===\n');
fprintf('평균 속도: %.2f m/s (%.2f km/h)\n', results.statistics.mean_speed, results.statistics.mean_speed*3.6);
fprintf('최대 속도: %.2f m/s (%.2f km/h)\n', results.statistics.max_speed, results.statistics.max_speed*3.6);
fprintf('속도 표준편차: %.2f m/s\n', results.statistics.std_speed);
fprintf('정지 비율: %.1f%%, 걷기 비율: %.1f%%, 달리기 비율: %.1f%%\n', ...
    results.statistics.stationary_ratio * 100, ...
    results.statistics.walking_ratio * 100, ...
    results.statistics.running_ratio * 100);

fprintf('\n분석 완료!\n');
