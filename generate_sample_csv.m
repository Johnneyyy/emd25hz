%% 다양한 샘플 CSV 파일 생성 스크립트
% 여러 유형의 테스트 신호를 CSV 파일로 생성
clear all; close all; clc;

fprintf('=== 샘플 CSV 파일 생성 ===\n\n');

%% 1. 기본 파라미터 설정
fs = 100;  % 샘플링 주파수 (Hz)
duration = 60;  % 신호 길이 (초)
t = (0:1/fs:duration-1/fs)';

%% 2. 다양한 신호 유형 생성

% 2.1 정상 신호 (Stationary signal)
signal_stationary = 0.5*sin(2*pi*0.5*t) + ...
                   0.3*sin(2*pi*1.2*t) + ...
                   0.4*sin(2*pi*2.5*t) + ...
                   0.2*sin(2*pi*3.8*t) + ...
                   0.1*randn(size(t));

% 2.2 주파수 변조 신호 (Chirp signal)
f0 = 0.5; f1 = 5.0;
signal_chirp = chirp(t, f0, duration, f1) + 0.05*randn(size(t));

% 2.3 시변 신호 (Time-varying signal)
signal_timevar = zeros(size(t));
for i = 1:length(t)
    if t(i) < 20
        signal_timevar(i) = sin(2*pi*0.8*t(i));
    elseif t(i) < 40
        signal_timevar(i) = sin(2*pi*2.0*t(i)) + 0.3*sin(2*pi*4.0*t(i));
    else
        signal_timevar(i) = sin(2*pi*3.5*t(i)) + 0.2*sin(2*pi*1.0*t(i));
    end
end
signal_timevar = signal_timevar + 0.05*randn(size(t));

% 2.4 펄스 신호 (Pulse signal)
signal_pulse = zeros(size(t));
pulse_times = [10, 20, 30, 40, 50];
pulse_width = 0.5;  % 초
for pt = pulse_times
    pulse_idx = find(abs(t - pt) < pulse_width/2);
    signal_pulse(pulse_idx) = 1;
end
% 펄스에 감쇠 추가
for i = 2:length(signal_pulse)
    if signal_pulse(i) == 0 && signal_pulse(i-1) > 0
        decay_length = round(fs * 2);  % 2초 감쇠
        for j = 1:min(decay_length, length(signal_pulse)-i)
            signal_pulse(i+j-1) = signal_pulse(i-1) * exp(-3*j/decay_length);
        end
    end
end
signal_pulse = signal_pulse + 0.02*randn(size(t));

% 2.5 AM 변조 신호 (Amplitude Modulated)
carrier_freq = 5.0;  % 반송파 주파수
modulation_freq = 0.2;  % 변조 주파수
modulation_index = 0.5;
signal_am = (1 + modulation_index*sin(2*pi*modulation_freq*t)) .* ...
           sin(2*pi*carrier_freq*t) + 0.05*randn(size(t));

% 2.6 생체 신호 유사 (Bio-signal like)
% 심박 유사 신호 생성
heart_rate = 70;  % BPM
heart_freq = heart_rate / 60;  % Hz
signal_bio = zeros(size(t));

% QRS 복합체 유사 펄스 생성
beat_times = 0:1/heart_freq:duration;
for bt = beat_times
    idx = find(abs(t - bt) < 0.05);
    if ~isempty(idx)
        % P파
        p_idx = find(abs(t - (bt - 0.1)) < 0.03);
        if ~isempty(p_idx)
            signal_bio(p_idx) = signal_bio(p_idx) + 0.2*exp(-((t(p_idx)-(bt-0.1))/0.02).^2);
        end
        % QRS 복합체
        signal_bio(idx) = signal_bio(idx) + exp(-((t(idx)-bt)/0.01).^2);
        % T파
        t_idx = find(abs(t - (bt + 0.15)) < 0.05);
        if ~isempty(t_idx)
            signal_bio(t_idx) = signal_bio(t_idx) + 0.3*exp(-((t(t_idx)-(bt+0.15))/0.03).^2);
        end
    end
end
% 호흡 변동 추가
respiration_freq = 0.25;  % Hz (15 breaths/min)
signal_bio = signal_bio .* (1 + 0.1*sin(2*pi*respiration_freq*t));
signal_bio = signal_bio + 0.02*randn(size(t));

%% 3. CSV 파일로 저장
signals = {signal_stationary, signal_chirp, signal_timevar, ...
          signal_pulse, signal_am, signal_bio};
names = {'stationary', 'chirp', 'timevarying', 'pulse', 'am_modulated', 'biosignal'};
descriptions = {
    '정상 신호 (0.5, 1.2, 2.5, 3.8 Hz)', ...
    '주파수 변조 신호 (0.5 → 5.0 Hz)', ...
    '시변 신호 (주파수가 시간에 따라 변함)', ...
    '펄스 신호 (감쇠 포함)', ...
    'AM 변조 신호 (반송파 5Hz, 변조 0.2Hz)', ...
    '생체 신호 유사 (심박 70BPM + 호흡 변동)'
};

for i = 1:length(signals)
    filename = sprintf('signal_%s.csv', names{i});
    data = [t, signals{i}'];
    writematrix(data, filename);
    fprintf('생성된 파일: %s\n', filename);
    fprintf('  설명: %s\n', descriptions{i});
    fprintf('  크기: %d x 2 (시간, 신호)\n\n', length(t));
end

%% 4. 다중 채널 신호 생성 (옵션)
fprintf('다중 채널 신호 생성...\n');

% 3채널 신호 생성
channel1 = 0.5*sin(2*pi*1.0*t) + 0.1*randn(size(t));
channel2 = 0.4*sin(2*pi*2.0*t) + 0.3*sin(2*pi*0.5*t) + 0.1*randn(size(t));
channel3 = chirp(t, 0.2, duration, 3.0) * 0.6 + 0.1*randn(size(t));

multi_channel_data = [t, channel1', channel2', channel3'];
writematrix(multi_channel_data, 'signal_multichannel.csv');
fprintf('생성된 파일: signal_multichannel.csv\n');
fprintf('  설명: 3채널 신호 (각 채널 다른 주파수 특성)\n');
fprintf('  크기: %d x 4 (시간, 채널1, 채널2, 채널3)\n\n', length(t));

%% 5. 메타데이터 파일 생성
metadata_file = 'signal_metadata.txt';
fid = fopen(metadata_file, 'w');
fprintf(fid, '=== 생성된 신호 파일 메타데이터 ===\n\n');
fprintf(fid, '생성 일시: %s\n', datestr(now));
fprintf(fid, '샘플링 주파수: %d Hz\n', fs);
fprintf(fid, '신호 길이: %d 초\n', duration);
fprintf(fid, '샘플 수: %d\n\n', length(t));

fprintf(fid, '파일 목록:\n');
for i = 1:length(names)
    fprintf(fid, '%d. signal_%s.csv\n', i, names{i});
    fprintf(fid, '   - %s\n', descriptions{i});
end
fprintf(fid, '\n%d. signal_multichannel.csv\n', length(names)+1);
fprintf(fid, '   - 3채널 신호 (각 채널 다른 주파수 특성)\n');

fclose(fid);
fprintf('메타데이터 파일 생성: %s\n\n', metadata_file);

%% 6. 간단한 시각화
figure('Position', [100, 100, 1400, 800]);
sgtitle('생성된 샘플 신호', 'FontSize', 14, 'FontWeight', 'bold');

for i = 1:6
    subplot(3, 2, i);
    plot(t(1:1000), signals{i}(1:1000), 'b-', 'LineWidth', 0.5);
    xlabel('시간 (초)');
    ylabel('진폭');
    title(sprintf('%s 신호', names{i}), 'Interpreter', 'none');
    grid on;
    xlim([0, 10]);
end

% 그래프 저장
saveas(gcf, 'sample_signals_preview.png');
fprintf('미리보기 이미지 저장: sample_signals_preview.png\n');

fprintf('\n=== 완료 ===\n');
fprintf('총 %d개의 CSV 파일이 생성되었습니다.\n', length(signals)+1);
fprintf('csv_signal_analysis.m 스크립트로 이 파일들을 분석할 수 있습니다.\n');