function [speed, velocity, filtered_acc, results] = calculate_movement_speed(acc_x, acc_y, acc_z, fs, varargin)
%% calculate_movement_speed - 스마트워치 가속도계 데이터로부터 이동 속도 계산
%
% 이 함수는 스마트워치에서 25Hz로 샘플링된 가속도계 데이터(x, y, z축)로부터
% 사람이 걷거나 달리거나, 워치를 보면서 뛰는 상황에서도 이동 속도를 계산합니다.
%
% 입력:
%   acc_x, acc_y, acc_z: 가속도계 데이터 (m/s^2 또는 g 단위)
%   fs: 샘플링 주파수 (Hz, 기본값: 25)
%   varargin: 선택적 파라미터
%     - 'gravity_removal': 중력 제거 방법 ('highpass', 'detrend', 'none')
%     - 'filter_type': 필터 타입 ('butter', 'cheby1', 'none')
%     - 'cutoff_freq': 고역 통과 필터 차단 주파수 (Hz, 기본값: 0.1)
%     - 'drift_correction': 드리프트 보정 방법 ('highpass', 'detrend', 'none')
%     - 'integration_window': 적분 윈도우 길이 (초, 기본값: 5)
%     - 'min_speed_threshold': 최소 속도 임계값 (m/s, 기본값: 0.1)
%     - 'calibration_time': 캘리브레이션 시간 (초, 기본값: 2)
%
% 출력:
%   speed: 이동 속도 (m/s)
%   velocity: 속도 벡터 (m/s)
%   filtered_acc: 필터링된 가속도 벡터 (m/s^2)
%   results: 상세 결과 구조체
%
% 사용 예시:
%   [speed, velocity, filtered_acc, results] = calculate_movement_speed(acc_x, acc_y, acc_z, 25);
%   [speed, velocity, filtered_acc, results] = calculate_movement_speed(acc_x, acc_y, acc_z, 25, ...
%       'gravity_removal', 'highpass', 'cutoff_freq', 0.2);
%
% 작성일: 2024
% 참고: 이 함수는 다양한 움직임 패턴(걷기, 달리기, 워치를 보면서 뛰기)을 처리합니다.

%% 1. 입력 파라미터 처리
p = inputParser;
addRequired(p, 'acc_x', @isnumeric);
addRequired(p, 'acc_y', @isnumeric);
addRequired(p, 'acc_z', @isnumeric);
addRequired(p, 'fs', @(x) isnumeric(x) && x > 0);
addParameter(p, 'gravity_removal', 'highpass', @(x) ismember(x, {'highpass', 'detrend', 'none'}));
addParameter(p, 'filter_type', 'butter', @(x) ismember(x, {'butter', 'cheby1', 'none'}));
addParameter(p, 'cutoff_freq', 0.1, @(x) isnumeric(x) && x > 0);
addParameter(p, 'drift_correction', 'highpass', @(x) ismember(x, {'highpass', 'detrend', 'none'}));
addParameter(p, 'integration_window', 5, @(x) isnumeric(x) && x > 0);
addParameter(p, 'min_speed_threshold', 0.1, @(x) isnumeric(x) && x >= 0);
addParameter(p, 'calibration_time', 2, @(x) isnumeric(x) && x > 0);
addParameter(p, 'unit', 'm/s2', @(x) ismember(x, {'m/s2', 'g'}));

parse(p, acc_x, acc_y, acc_z, fs, varargin{:});

% 파라미터 추출
gravity_removal = p.Results.gravity_removal;
filter_type = p.Results.filter_type;
cutoff_freq = p.Results.cutoff_freq;
drift_correction = p.Results.drift_correction;
integration_window = p.Results.integration_window;
min_speed_threshold = p.Results.min_speed_threshold;
calibration_time = p.Results.calibration_time;
unit = p.Results.unit;

% 데이터 길이 확인
if length(acc_x) ~= length(acc_y) || length(acc_x) ~= length(acc_z)
    error('가속도계 데이터의 길이가 일치하지 않습니다.');
end

N = length(acc_x);
dt = 1/fs;
t = (0:N-1) * dt;

% 단위 변환 (g를 m/s^2로 변환)
if strcmpi(unit, 'g')
    acc_x = acc_x * 9.81;
    acc_y = acc_y * 9.81;
    acc_z = acc_z * 9.81;
end

fprintf('가속도계 데이터 처리 시작...\n');
fprintf('데이터 길이: %d 샘플, 샘플링 주파수: %.1f Hz, 기간: %.2f 초\n', N, fs, t(end));

%% 2. 중력 성분 제거
fprintf('중력 성분 제거 중...\n');

switch gravity_removal
    case 'highpass'
        % 고역 통과 필터를 사용하여 중력 성분 제거
        nyquist = fs / 2;
        if cutoff_freq >= nyquist
            warning('차단 주파수가 나이퀴스트 주파수보다 큽니다. 차단 주파수를 조정합니다.');
            cutoff_freq = nyquist * 0.9;
        end
        
        [b, a] = butter(4, cutoff_freq/nyquist, 'high');
        acc_x_no_gravity = filtfilt(b, a, acc_x);
        acc_y_no_gravity = filtfilt(b, a, acc_y);
        acc_z_no_gravity = filtfilt(b, a, acc_z);
        
    case 'detrend'
        % 선형 추세 제거
        acc_x_no_gravity = detrend(acc_x);
        acc_y_no_gravity = detrend(acc_y);
        acc_z_no_gravity = detrend(acc_z);
        
    case 'none'
        acc_x_no_gravity = acc_x;
        acc_y_no_gravity = acc_y;
        acc_z_no_gravity = acc_z;
end

%% 3. 노이즈 필터링
fprintf('노이즈 필터링 중...\n');

if ~strcmpi(filter_type, 'none')
    % 이동 평균 필터 (짧은 윈도우)
    window_samples = max(1, round(0.1 * fs)); % 0.1초 윈도우
    acc_x_filtered = movmean(acc_x_no_gravity, window_samples);
    acc_y_filtered = movmean(acc_y_no_gravity, window_samples);
    acc_z_filtered = movmean(acc_z_no_gravity, window_samples);
else
    acc_x_filtered = acc_x_no_gravity;
    acc_y_filtered = acc_y_no_gravity;
    acc_z_filtered = acc_z_no_gravity;
end

%% 4. 가속도 벡터 크기 계산 (움직임 강도)
fprintf('가속도 벡터 크기 계산 중...\n');

% 전체 가속도 벡터 크기
acc_magnitude = sqrt(acc_x_filtered.^2 + acc_y_filtered.^2 + acc_z_filtered.^2);

% 정적 가속도 제거 (중력 제거 후의 평균값)
calibration_samples = round(calibration_time * fs);
if calibration_samples < N
    static_acc = mean(acc_magnitude(1:calibration_samples));
else
    static_acc = mean(acc_magnitude);
end

acc_dynamic = acc_magnitude - static_acc;

%% 5. 움직임 감지 및 활동 분류
fprintf('움직임 감지 및 활동 분류 중...\n');

% 움직임 에너지 계산 (슬라이딩 윈도우)
window_size = round(1.0 * fs); % 1초 윈도우
movement_energy = movmean(acc_dynamic.^2, window_size);

% 활동 분류 임계값
walking_threshold = 0.5; % m^2/s^4 (조정 필요)
running_threshold = 2.0; % m^2/s^4 (조정 필요)

activity_type = zeros(N, 1); % 0: 정지, 1: 걷기, 2: 달리기
for i = 1:N
    if movement_energy(i) < walking_threshold
        activity_type(i) = 0; % 정지
    elseif movement_energy(i) < running_threshold
        activity_type(i) = 1; % 걷기
    else
        activity_type(i) = 2; % 달리기
    end
end

%% 6. 가속도 적분을 통한 속도 계산
fprintf('가속도 적분을 통한 속도 계산 중...\n');

% 적분 윈도우 길이
window_samples = round(integration_window * fs);

% 슬라이딩 윈도우 적분
velocity_x = zeros(N, 1);
velocity_y = zeros(N, 1);
velocity_z = zeros(N, 1);

for i = 1:N
    start_idx = max(1, i - window_samples + 1);
    end_idx = i;
    
    % 적분 (사다리꼴 법칙)
    if end_idx > start_idx
        velocity_x(i) = trapz(t(start_idx:end_idx), acc_x_filtered(start_idx:end_idx));
        velocity_y(i) = trapz(t(start_idx:end_idx), acc_y_filtered(start_idx:end_idx));
        velocity_z(i) = trapz(t(start_idx:end_idx), acc_z_filtered(start_idx:end_idx));
    end
end

%% 7. 드리프트 보정
fprintf('드리프트 보정 중...\n');

switch drift_correction
    case 'highpass'
        % 고역 통과 필터로 저주파 드리프트 제거
        drift_cutoff = 0.05; % Hz
        nyquist = fs / 2;
        if drift_cutoff < nyquist
            [b_drift, a_drift] = butter(2, drift_cutoff/nyquist, 'high');
            velocity_x = filtfilt(b_drift, a_drift, velocity_x);
            velocity_y = filtfilt(b_drift, a_drift, velocity_y);
            velocity_z = filtfilt(b_drift, a_drift, velocity_z);
        end
        
    case 'detrend'
        % 선형 추세 제거
        velocity_x = detrend(velocity_x);
        velocity_y = detrend(velocity_y);
        velocity_z = detrend(velocity_z);
        
    case 'none'
        % 드리프트 보정 없음
end

%% 8. 속도 크기 계산 및 후처리
fprintf('속도 크기 계산 및 후처리 중...\n');

% 속도 벡터 크기
velocity_magnitude = sqrt(velocity_x.^2 + velocity_y.^2 + velocity_z.^2);

% 음수 속도 제거 (물리적으로 불가능)
velocity_magnitude(velocity_magnitude < 0) = 0;

% 최소 속도 임계값 적용
velocity_magnitude(velocity_magnitude < min_speed_threshold) = 0;

% 이동 평균으로 부드럽게 만들기
smooth_window = round(0.5 * fs); % 0.5초 윈도우
speed = movmean(velocity_magnitude, smooth_window);

%% 9. 결과 구조체 생성
results = struct();
results.time = t;
results.acceleration.raw.x = acc_x;
results.acceleration.raw.y = acc_y;
results.acceleration.raw.z = acc_z;
results.acceleration.filtered.x = acc_x_filtered;
results.acceleration.filtered.y = acc_y_filtered;
results.acceleration.filtered.z = acc_z_filtered;
results.acceleration.magnitude = acc_magnitude;
results.acceleration.dynamic = acc_dynamic;
results.velocity.x = velocity_x;
results.velocity.y = velocity_y;
results.velocity.z = velocity_z;
results.velocity.magnitude = velocity_magnitude;
results.speed = speed;
results.activity_type = activity_type;
results.movement_energy = movement_energy;
results.parameters.gravity_removal = gravity_removal;
results.parameters.filter_type = filter_type;
results.parameters.cutoff_freq = cutoff_freq;
results.parameters.drift_correction = drift_correction;
results.parameters.integration_window = integration_window;
results.parameters.min_speed_threshold = min_speed_threshold;
results.parameters.calibration_time = calibration_time;
results.parameters.fs = fs;

% 통계 정보
results.statistics.mean_speed = mean(speed);
results.statistics.max_speed = max(speed);
results.statistics.std_speed = std(speed);
results.statistics.mean_acceleration = mean(acc_magnitude);
results.statistics.max_acceleration = max(acc_magnitude);

% 활동 분류 통계
results.statistics.stationary_ratio = sum(activity_type == 0) / N;
results.statistics.walking_ratio = sum(activity_type == 1) / N;
results.statistics.running_ratio = sum(activity_type == 2) / N;

fprintf('\n=== 속도 계산 완료 ===\n');
fprintf('평균 속도: %.2f m/s (%.2f km/h)\n', results.statistics.mean_speed, results.statistics.mean_speed * 3.6);
fprintf('최대 속도: %.2f m/s (%.2f km/h)\n', results.statistics.max_speed, results.statistics.max_speed * 3.6);
fprintf('정지 비율: %.1f%%, 걷기 비율: %.1f%%, 달리기 비율: %.1f%%\n', ...
    results.statistics.stationary_ratio * 100, ...
    results.statistics.walking_ratio * 100, ...
    results.statistics.running_ratio * 100);

filtered_acc = [acc_x_filtered, acc_y_filtered, acc_z_filtered];
velocity = [velocity_x, velocity_y, velocity_z];

end
