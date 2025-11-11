function [acc_x, acc_y, acc_z, true_speed] = generate_sample_accelerometer_data(fs, duration, activity_type, varargin)
%% generate_sample_accelerometer_data - 샘플 가속도계 데이터 생성
%
% 이 함수는 테스트를 위한 샘플 가속도계 데이터를 생성합니다.
%
% 입력:
%   fs: 샘플링 주파수 (Hz, 기본값: 25)
%   duration: 신호 길이 (초, 기본값: 30)
%   activity_type: 활동 유형 ('walking', 'running', 'mixed', 'watch_viewing')
%   varargin: 선택적 파라미터
%     - 'walking_speed': 걷기 속도 (m/s, 기본값: 1.4)
%     - 'running_speed': 달리기 속도 (m/s, 기본값: 3.0)
%     - 'noise_level': 노이즈 레벨 (기본값: 0.1)
%
% 출력:
%   acc_x, acc_y, acc_z: 가속도계 데이터 (m/s^2)
%   true_speed: 실제 속도 (m/s) - 검증용

p = inputParser;
addRequired(p, 'fs', @isnumeric);
addRequired(p, 'duration', @isnumeric);
addRequired(p, 'activity_type', @(x) ismember(x, {'walking', 'running', 'mixed', 'watch_viewing'}));
addParameter(p, 'walking_speed', 1.4, @isnumeric);
addParameter(p, 'running_speed', 3.0, @isnumeric);
addParameter(p, 'noise_level', 0.1, @isnumeric);

parse(p, fs, duration, varargin{:});

walking_speed = p.Results.walking_speed;
running_speed = p.Results.running_speed;
noise_level = p.Results.noise_level;

N = round(fs * duration);
t = (0:N-1) / fs;

% 중력 벡터 (기본적으로 z축 방향)
gravity = [0, 0, 9.81]; % m/s^2

% 초기화
acc_x = zeros(N, 1);
acc_y = zeros(N, 1);
acc_z = zeros(N, 1);
true_speed = zeros(N, 1);

switch activity_type
    case 'walking'
        % 걷기: 주기적인 가속도 변화
        step_freq = 2.0; % Hz (초당 2보)
        for i = 1:N
            phase = 2 * pi * step_freq * t(i);
            % 주기적인 가속도 변화
            acc_magnitude = 2.0 * (1 + 0.5 * sin(phase));
            acc_x(i) = acc_magnitude * sin(phase) + noise_level * randn;
            acc_y(i) = acc_magnitude * cos(phase) + noise_level * randn;
            acc_z(i) = gravity(3) + acc_magnitude * 0.3 * sin(phase + pi/4) + noise_level * randn;
            true_speed(i) = walking_speed * (1 + 0.1 * sin(phase));
        end
        
    case 'running'
        % 달리기: 더 빠르고 강한 가속도 변화
        step_freq = 3.0; % Hz (초당 3보)
        for i = 1:N
            phase = 2 * pi * step_freq * t(i);
            acc_magnitude = 5.0 * (1 + 0.6 * sin(phase));
            acc_x(i) = acc_magnitude * sin(phase) + noise_level * randn;
            acc_y(i) = acc_magnitude * cos(phase) + noise_level * randn;
            acc_z(i) = gravity(3) + acc_magnitude * 0.5 * sin(phase + pi/4) + noise_level * randn;
            true_speed(i) = running_speed * (1 + 0.15 * sin(phase));
        end
        
    case 'mixed'
        % 혼합: 걷기와 달리기 교차
        for i = 1:N
            if mod(floor(t(i) / 5), 2) == 0
                % 걷기 구간
                step_freq = 2.0;
                phase = 2 * pi * step_freq * t(i);
                acc_magnitude = 2.0 * (1 + 0.5 * sin(phase));
                acc_x(i) = acc_magnitude * sin(phase) + noise_level * randn;
                acc_y(i) = acc_magnitude * cos(phase) + noise_level * randn;
                acc_z(i) = gravity(3) + acc_magnitude * 0.3 * sin(phase + pi/4) + noise_level * randn;
                true_speed(i) = walking_speed * (1 + 0.1 * sin(phase));
            else
                % 달리기 구간
                step_freq = 3.0;
                phase = 2 * pi * step_freq * t(i);
                acc_magnitude = 5.0 * (1 + 0.6 * sin(phase));
                acc_x(i) = acc_magnitude * sin(phase) + noise_level * randn;
                acc_y(i) = acc_magnitude * cos(phase) + noise_level * randn;
                acc_z(i) = gravity(3) + acc_magnitude * 0.5 * sin(phase + pi/4) + noise_level * randn;
                true_speed(i) = running_speed * (1 + 0.15 * sin(phase));
            end
        end
        
    case 'watch_viewing'
        % 워치를 보면서 뛰기: 추가적인 회전 운동
        step_freq = 3.0;
        for i = 1:N
            phase = 2 * pi * step_freq * t(i);
            acc_magnitude = 5.0 * (1 + 0.6 * sin(phase));
            
            % 워치 회전 효과 (주기적인 회전)
            rotation_phase = 2 * pi * 0.5 * t(i); % 0.5 Hz 회전
            rot_matrix = [cos(rotation_phase), -sin(rotation_phase), 0;
                         sin(rotation_phase), cos(rotation_phase), 0;
                         0, 0, 1];
            
            acc_vec = [acc_magnitude * sin(phase);
                      acc_magnitude * cos(phase);
                      acc_magnitude * 0.5 * sin(phase + pi/4)];
            
            acc_rotated = rot_matrix * acc_vec;
            
            acc_x(i) = acc_rotated(1) + noise_level * randn;
            acc_y(i) = acc_rotated(2) + noise_level * randn;
            acc_z(i) = gravity(3) + acc_rotated(3) + noise_level * randn;
            true_speed(i) = running_speed * (1 + 0.15 * sin(phase));
        end
end

end
