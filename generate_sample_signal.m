%% 예제 신호 CSV 파일 생성
% 테스트용 신호를 생성하여 CSV 파일로 저장
clear all; close all; clc;

%% 1. 신호 생성 파라미터
fs = 100;  % 샘플링 주파수 (Hz)
duration = 60;  % 신호 길이 (초)
t = 0:1/fs:duration-1/fs;

%% 2. 다양한 테스트 신호 생성
% 2.1 복합 주파수 신호
signal1 = 0.5*sin(2*pi*0.5*t) + ...    % 0.5 Hz
          0.3*sin(2*pi*1.2*t) + ...    % 1.2 Hz
          0.4*sin(2*pi*2.5*t) + ...    % 2.5 Hz
          0.2*sin(2*pi*3.8*t) + ...    % 3.8 Hz
          0.1*randn(size(t));          % 잡음

% 2.2 시변 신호
signal2 = zeros(size(t));
for i = 1:length(t)
    if t(i) < 20
        signal2(i) = 0.6*sin(2*pi*0.8*t(i));
    elseif t(i) < 40
        signal2(i) = 0.5*sin(2*pi*2.0*t(i));
    else
        signal2(i) = 0.4*sin(2*pi*3.5*t(i));
    end
end
signal2 = signal2 + 0.05*randn(size(t));

% 2.3 주파수 변조 신호 (Chirp)
f0 = 0.5; f1 = 3.5;
signal3 = chirp(t, f0, duration, f1) + 0.1*randn(size(t));

%% 3. CSV 파일로 저장
% 형식: 시간, 신호값
data1 = [t', signal1'];
data2 = [t', signal2'];
data3 = [t', signal3'];

% CSV 파일 저장
csvwrite('test_signal_complex.csv', data1);
csvwrite('test_signal_timevarying.csv', data2);
csvwrite('test_signal_chirp.csv', data3);

% 헤더가 있는 버전도 생성
fid = fopen('test_signal_with_header.csv', 'w');
fprintf(fid, 'Time(s),Signal\n');
for i = 1:length(t)
    fprintf(fid, '%.4f,%.6f\n', t(i), signal1(i));
end
fclose(fid);

%% 4. 생성된 신호 확인
figure('Position', [100, 100, 1200, 800]);

subplot(3,2,1);
plot(t, signal1);
xlabel('시간 (초)');
ylabel('진폭');
title('복합 주파수 신호');
grid on;

subplot(3,2,2);
Y1 = fft(signal1);
f = fs*(0:(length(signal1)/2))/length(signal1);
P1 = abs(Y1/length(signal1));
P1 = P1(1:length(signal1)/2+1);
P1(2:end-1) = 2*P1(2:end-1);
plot(f, P1);
xlabel('주파수 (Hz)');
ylabel('진폭');
title('복합 주파수 신호 스펙트럼');
xlim([0, 5]);
grid on;

subplot(3,2,3);
plot(t, signal2);
xlabel('시간 (초)');
ylabel('진폭');
title('시변 신호');
grid on;

subplot(3,2,4);
spectrogram(signal2, hamming(256), 250, 512, fs, 'yaxis');
title('시변 신호 스펙트로그램');
ylim([0, 5]);

subplot(3,2,5);
plot(t, signal3);
xlabel('시간 (초)');
ylabel('진폭');
title('주파수 변조 신호 (Chirp)');
grid on;

subplot(3,2,6);
spectrogram(signal3, hamming(256), 250, 512, fs, 'yaxis');
title('Chirp 신호 스펙트로그램');
ylim([0, 5]);

fprintf('CSV 파일이 생성되었습니다:\n');
fprintf('1. test_signal_complex.csv - 복합 주파수 신호\n');
fprintf('2. test_signal_timevarying.csv - 시변 신호\n');
fprintf('3. test_signal_chirp.csv - 주파수 변조 신호\n');
fprintf('4. test_signal_with_header.csv - 헤더가 있는 복합 주파수 신호\n');