%% 워치 스트랩 차이에 따른 가속도계 센서 미세 진동 차이 분석
% 25Hz 샘플링 주파수의 ACC X/Y/Z 신호 분석
% Author: MATLAB Signal Processing
% Date: 2024

clear all; close all; clc;

%% 1. 파라미터 설정
fs = 25;                    % 샘플링 주파수 (Hz)
duration = 60;              % 측정 시간 (초) - 1분
t = 0:1/fs:duration-1/fs;    % 시간 벡터
N = length(t);

fprintf('=== 워치 스트랩 차이에 따른 ACC 센서 미세 진동 분석 ===\n');
fprintf('샘플링 주파수: %d Hz\n', fs);
fprintf('측정 시간: %.1f 초\n', duration);
fprintf('총 샘플 수: %d\n\n', N);

%% 2. 샘플 데이터 생성 (실제 데이터로 대체 가능)
% 스트랩 A: 더 타이트한 스트랩 (미세 진동이 적음)
% 스트랩 B: 더 느슨한 스트랩 (미세 진동이 많음)

% 기본 움직임 신호 (사용자 활동)
base_freq = 0.5;            % 기본 움직임 주파수 (Hz)
base_amplitude = 0.5;       % 기본 움직임 진폭

% 스트랩 A 데이터 생성
noise_level_A = 0.02;       % 낮은 노이즈 레벨 (타이트한 스트랩)
high_freq_noise_A = 0.01;   % 고주파 미세 진동 (낮음)

acc_x_A = base_amplitude * sin(2*pi*base_freq*t) + ...
          noise_level_A * randn(size(t)) + ...
          high_freq_noise_A * sin(2*pi*8*t + rand(size(t))) + ...
          high_freq_noise_A * 0.5 * sin(2*pi*12*t + rand(size(t)));

acc_y_A = base_amplitude * cos(2*pi*base_freq*t) + ...
          noise_level_A * randn(size(t)) + ...
          high_freq_noise_A * sin(2*pi*8*t + rand(size(t))) + ...
          high_freq_noise_A * 0.5 * sin(2*pi*12*t + rand(size(t)));

acc_z_A = base_amplitude * 0.7 * sin(2*pi*base_freq*t + pi/4) + ...
          noise_level_A * randn(size(t)) + ...
          high_freq_noise_A * sin(2*pi*8*t + rand(size(t))) + ...
          high_freq_noise_A * 0.5 * sin(2*pi*12*t + rand(size(t)));

% 스트랩 B 데이터 생성
noise_level_B = 0.05;       % 높은 노이즈 레벨 (느슨한 스트랩)
high_freq_noise_B = 0.03;   % 고주파 미세 진동 (높음)

acc_x_B = base_amplitude * sin(2*pi*base_freq*t) + ...
          noise_level_B * randn(size(t)) + ...
          high_freq_noise_B * sin(2*pi*8*t + rand(size(t))) + ...
          high_freq_noise_B * 0.5 * sin(2*pi*12*t + rand(size(t))) + ...
          high_freq_noise_B * 0.3 * sin(2*pi*15*t + rand(size(t)));

acc_y_B = base_amplitude * cos(2*pi*base_freq*t) + ...
          noise_level_B * randn(size(t)) + ...
          high_freq_noise_B * sin(2*pi*8*t + rand(size(t))) + ...
          high_freq_noise_B * 0.5 * sin(2*pi*12*t + rand(size(t))) + ...
          high_freq_noise_B * 0.3 * sin(2*pi*15*t + rand(size(t)));

acc_z_B = base_amplitude * 0.7 * sin(2*pi*base_freq*t + pi/4) + ...
          noise_level_B * randn(size(t)) + ...
          high_freq_noise_B * sin(2*pi*8*t + rand(size(t))) + ...
          high_freq_noise_B * 0.5 * sin(2*pi*12*t + rand(size(t))) + ...
          high_freq_noise_B * 0.3 * sin(2*pi*15*t + rand(size(t)));

% 실제 데이터를 사용하는 경우 아래 주석을 해제하고 위의 생성 코드를 주석 처리
% load('strap_A_data.mat');  % acc_x_A, acc_y_A, acc_z_A 변수 포함
% load('strap_B_data.mat');  % acc_x_B, acc_y_B, acc_z_B 변수 포함

%% 3. 통계적 분석
fprintf('=== 통계적 분석 결과 ===\n');

% RMS (Root Mean Square) 값 계산
rms_x_A = sqrt(mean(acc_x_A.^2));
rms_y_A = sqrt(mean(acc_y_A.^2));
rms_z_A = sqrt(mean(acc_z_A.^2));
rms_total_A = sqrt(mean(acc_x_A.^2 + acc_y_A.^2 + acc_z_A.^2));

rms_x_B = sqrt(mean(acc_x_B.^2));
rms_y_B = sqrt(mean(acc_y_B.^2));
rms_z_B = sqrt(mean(acc_z_B.^2));
rms_total_B = sqrt(mean(acc_x_B.^2 + acc_y_B.^2 + acc_z_B.^2));

% 표준편차 계산
std_x_A = std(acc_x_A);
std_y_A = std(acc_y_A);
std_z_A = std(acc_z_A);
std_total_A = std(sqrt(acc_x_A.^2 + acc_y_A.^2 + acc_z_A.^2));

std_x_B = std(acc_x_B);
std_y_B = std(acc_y_B);
std_z_B = std(acc_z_B);
std_total_B = std(sqrt(acc_x_B.^2 + acc_y_B.^2 + acc_z_B.^2));

% 변동계수 (Coefficient of Variation)
cv_x_A = std_x_A / abs(mean(acc_x_A));
cv_y_A = std_y_A / abs(mean(acc_y_A));
cv_z_A = std_z_A / abs(mean(acc_z_A));

cv_x_B = std_x_B / abs(mean(acc_x_B));
cv_y_B = std_y_B / abs(mean(acc_y_B));
cv_z_B = std_z_B / abs(mean(acc_z_B));

fprintf('\n--- RMS 값 비교 ---\n');
fprintf('스트랩 A: X=%.4f, Y=%.4f, Z=%.4f, Total=%.4f\n', ...
        rms_x_A, rms_y_A, rms_z_A, rms_total_A);
fprintf('스트랩 B: X=%.4f, Y=%.4f, Z=%.4f, Total=%.4f\n', ...
        rms_x_B, rms_y_B, rms_z_B, rms_total_B);
fprintf('차이 (B-A): X=%.4f (%.1f%%), Y=%.4f (%.1f%%), Z=%.4f (%.1f%%), Total=%.4f (%.1f%%)\n', ...
        rms_x_B-rms_x_A, 100*(rms_x_B-rms_x_A)/rms_x_A, ...
        rms_y_B-rms_y_A, 100*(rms_y_B-rms_y_A)/rms_y_A, ...
        rms_z_B-rms_z_A, 100*(rms_z_B-rms_z_A)/rms_z_A, ...
        rms_total_B-rms_total_A, 100*(rms_total_B-rms_total_A)/rms_total_A);

fprintf('\n--- 표준편차 비교 ---\n');
fprintf('스트랩 A: X=%.4f, Y=%.4f, Z=%.4f, Total=%.4f\n', ...
        std_x_A, std_y_A, std_z_A, std_total_A);
fprintf('스트랩 B: X=%.4f, Y=%.4f, Z=%.4f, Total=%.4f\n', ...
        std_x_B, std_y_B, std_z_B, std_total_B);
fprintf('차이 (B-A): X=%.4f (%.1f%%), Y=%.4f (%.1f%%), Z=%.4f (%.1f%%), Total=%.4f (%.1f%%)\n', ...
        std_x_B-std_x_A, 100*(std_x_B-std_x_A)/std_x_A, ...
        std_y_B-std_y_A, 100*(std_y_B-std_y_A)/std_y_A, ...
        std_z_B-std_z_A, 100*(std_z_B-std_z_A)/std_z_A, ...
        std_total_B-std_total_A, 100*(std_total_B-std_total_A)/std_total_A);

%% 4. 주파수 도메인 분석
fprintf('\n=== 주파수 도메인 분석 ===\n');

% FFT 분석
N_fft = 2^nextpow2(N);
freq = fs * (0:(N_fft/2)) / N_fft;

% 스트랩 A FFT
fft_x_A = fft(acc_x_A, N_fft);
fft_y_A = fft(acc_y_A, N_fft);
fft_z_A = fft(acc_z_A, N_fft);
Pxx_x_A = abs(fft_x_A(1:N_fft/2+1)).^2 / N_fft;
Pxx_y_A = abs(fft_y_A(1:N_fft/2+1)).^2 / N_fft;
Pxx_z_A = abs(fft_z_A(1:N_fft/2+1)).^2 / N_fft;

% 스트랩 B FFT
fft_x_B = fft(acc_x_B, N_fft);
fft_y_B = fft(acc_y_B, N_fft);
fft_z_B = fft(acc_z_B, N_fft);
Pxx_x_B = abs(fft_x_B(1:N_fft/2+1)).^2 / N_fft;
Pxx_y_B = abs(fft_y_B(1:N_fft/2+1)).^2 / N_fft;
Pxx_z_B = abs(fft_z_B(1:N_fft/2+1)).^2 / N_fft;

% 고주파 성분 분석 (미세 진동: 5Hz 이상)
high_freq_idx = freq >= 5;
high_freq_power_x_A = sum(Pxx_x_A(high_freq_idx));
high_freq_power_y_A = sum(Pxx_y_A(high_freq_idx));
high_freq_power_z_A = sum(Pxx_z_A(high_freq_idx));
high_freq_power_total_A = high_freq_power_x_A + high_freq_power_y_A + high_freq_power_z_A;

high_freq_power_x_B = sum(Pxx_x_B(high_freq_idx));
high_freq_power_y_B = sum(Pxx_y_B(high_freq_idx));
high_freq_power_z_B = sum(Pxx_z_B(high_freq_idx));
high_freq_power_total_B = high_freq_power_x_B + high_freq_power_y_B + high_freq_power_z_B;

fprintf('\n--- 고주파 성분 파워 (5Hz 이상, 미세 진동) ---\n');
fprintf('스트랩 A: X=%.6f, Y=%.6f, Z=%.6f, Total=%.6f\n', ...
        high_freq_power_x_A, high_freq_power_y_A, high_freq_power_z_A, high_freq_power_total_A);
fprintf('스트랩 B: X=%.6f, Y=%.6f, Z=%.6f, Total=%.6f\n', ...
        high_freq_power_x_B, high_freq_power_y_B, high_freq_power_z_B, high_freq_power_total_B);
fprintf('차이 (B-A): X=%.6f (%.1f%%), Y=%.6f (%.1f%%), Z=%.6f (%.1f%%), Total=%.6f (%.1f%%)\n', ...
        high_freq_power_x_B-high_freq_power_x_A, 100*(high_freq_power_x_B-high_freq_power_x_A)/high_freq_power_x_A, ...
        high_freq_power_y_B-high_freq_power_y_A, 100*(high_freq_power_y_B-high_freq_power_y_A)/high_freq_power_y_A, ...
        high_freq_power_z_B-high_freq_power_z_A, 100*(high_freq_power_z_B-high_freq_power_z_A)/high_freq_power_z_A, ...
        high_freq_power_total_B-high_freq_power_total_A, 100*(high_freq_power_total_B-high_freq_power_total_A)/high_freq_power_total_A);

% 주파수 대역별 에너지 분석
freq_bands = [0, 1, 2, 5, 10, 12.5];  % Hz
fprintf('\n--- 주파수 대역별 에너지 분포 ---\n');
for i = 1:length(freq_bands)-1
    band_idx = (freq >= freq_bands(i)) & (freq < freq_bands(i+1));
    band_power_A = sum(Pxx_x_A(band_idx) + Pxx_y_A(band_idx) + Pxx_z_A(band_idx));
    band_power_B = sum(Pxx_x_B(band_idx) + Pxx_y_B(band_idx) + Pxx_z_B(band_idx));
    fprintf('%.1f-%.1f Hz: 스트랩 A=%.6f, 스트랩 B=%.6f, 차이=%.6f (%.1f%%)\n', ...
            freq_bands(i), freq_bands(i+1), band_power_A, band_power_B, ...
            band_power_B-band_power_A, 100*(band_power_B-band_power_A)/band_power_A);
end

%% 5. 시각화
fprintf('\n=== 시각화 생성 중... ===\n');

% Figure 1: 시간 도메인 비교
figure('Position', [100, 100, 1400, 900]);

% X축 신호
subplot(3,3,1);
plot(t(1:min(500, N)), acc_x_A(1:min(500, N)), 'b-', 'LineWidth', 1.5); hold on;
plot(t(1:min(500, N)), acc_x_B(1:min(500, N)), 'r--', 'LineWidth', 1.5);
title('ACC X축 신호 비교 (첫 20초)');
xlabel('시간 (s)'); ylabel('가속도 (g)');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;

subplot(3,3,2);
plot(t(1:min(500, N)), acc_y_A(1:min(500, N)), 'b-', 'LineWidth', 1.5); hold on;
plot(t(1:min(500, N)), acc_y_B(1:min(500, N)), 'r--', 'LineWidth', 1.5);
title('ACC Y축 신호 비교 (첫 20초)');
xlabel('시간 (s)'); ylabel('가속도 (g)');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;

subplot(3,3,3);
plot(t(1:min(500, N)), acc_z_A(1:min(500, N)), 'b-', 'LineWidth', 1.5); hold on;
plot(t(1:min(500, N)), acc_z_B(1:min(500, N)), 'r--', 'LineWidth', 1.5);
title('ACC Z축 신호 비교 (첫 20초)');
xlabel('시간 (s)'); ylabel('가속도 (g)');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;

% 주파수 스펙트럼 비교
subplot(3,3,4);
semilogy(freq, Pxx_x_A, 'b-', 'LineWidth', 1.5); hold on;
semilogy(freq, Pxx_x_B, 'r--', 'LineWidth', 1.5);
title('ACC X축 파워 스펙트럼');
xlabel('주파수 (Hz)'); ylabel('파워');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;
xlim([0 fs/2]);

subplot(3,3,5);
semilogy(freq, Pxx_y_A, 'b-', 'LineWidth', 1.5); hold on;
semilogy(freq, Pxx_y_B, 'r--', 'LineWidth', 1.5);
title('ACC Y축 파워 스펙트럼');
xlabel('주파수 (Hz)'); ylabel('파워');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;
xlim([0 fs/2]);

subplot(3,3,6);
semilogy(freq, Pxx_z_A, 'b-', 'LineWidth', 1.5); hold on;
semilogy(freq, Pxx_z_B, 'r--', 'LineWidth', 1.5);
title('ACC Z축 파워 스펙트럼');
xlabel('주파수 (Hz)'); ylabel('파워');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;
xlim([0 fs/2]);

% 고주파 성분 확대 (미세 진동 분석)
subplot(3,3,7);
high_freq_range = freq >= 5 & freq <= fs/2;
semilogy(freq(high_freq_range), Pxx_x_A(high_freq_range), 'b-', 'LineWidth', 2); hold on;
semilogy(freq(high_freq_range), Pxx_x_B(high_freq_range), 'r--', 'LineWidth', 2);
title('X축 고주파 성분 (5Hz 이상, 미세 진동)');
xlabel('주파수 (Hz)'); ylabel('파워');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;

subplot(3,3,8);
semilogy(freq(high_freq_range), Pxx_y_A(high_freq_range), 'b-', 'LineWidth', 2); hold on;
semilogy(freq(high_freq_range), Pxx_y_B(high_freq_range), 'r--', 'LineWidth', 2);
title('Y축 고주파 성분 (5Hz 이상, 미세 진동)');
xlabel('주파수 (Hz)'); ylabel('파워');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;

subplot(3,3,9);
semilogy(freq(high_freq_range), Pxx_z_A(high_freq_range), 'b-', 'LineWidth', 2); hold on;
semilogy(freq(high_freq_range), Pxx_z_B(high_freq_range), 'r--', 'LineWidth', 2);
title('Z축 고주파 성분 (5Hz 이상, 미세 진동)');
xlabel('주파수 (Hz)'); ylabel('파워');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;

sgtitle('워치 스트랩 차이에 따른 ACC 센서 미세 진동 분석', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 2: 통계적 비교 및 스펙트로그램
figure('Position', [100, 50, 1400, 900]);

% RMS 비교
subplot(3,3,1);
bar_data = [rms_x_A, rms_y_A, rms_z_A, rms_total_A; ...
            rms_x_B, rms_y_B, rms_z_B, rms_total_B]';
bar(bar_data);
set(gca, 'XTickLabel', {'X', 'Y', 'Z', 'Total'});
ylabel('RMS 값');
title('RMS 값 비교');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;

% 표준편차 비교
subplot(3,3,2);
bar_data = [std_x_A, std_y_A, std_z_A, std_total_A; ...
            std_x_B, std_y_B, std_z_B, std_total_B]';
bar(bar_data);
set(gca, 'XTickLabel', {'X', 'Y', 'Z', 'Total'});
ylabel('표준편차');
title('표준편차 비교');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;

% 고주파 파워 비교
subplot(3,3,3);
bar_data = [high_freq_power_x_A, high_freq_power_y_A, high_freq_power_z_A, high_freq_power_total_A; ...
            high_freq_power_x_B, high_freq_power_y_B, high_freq_power_z_B, high_freq_power_total_B]';
bar(bar_data);
set(gca, 'XTickLabel', {'X', 'Y', 'Z', 'Total'});
ylabel('고주파 파워 (5Hz 이상)');
title('고주파 성분 파워 비교');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;

% 스펙트로그램 - X축
subplot(3,3,4);
[s_A_x, f_spec, t_spec] = spectrogram(acc_x_A, round(N/10), round(N/20), [], fs);
imagesc(t_spec, f_spec, 10*log10(abs(s_A_x).^2));
axis xy; colorbar;
title('스트랩 A - X축 스펙트로그램');
xlabel('시간 (s)'); ylabel('주파수 (Hz)');
ylim([0 fs/2]);

subplot(3,3,5);
[s_B_x, ~, ~] = spectrogram(acc_x_B, round(N/10), round(N/20), [], fs);
imagesc(t_spec, f_spec, 10*log10(abs(s_B_x).^2));
axis xy; colorbar;
title('스트랩 B - X축 스펙트로그램');
xlabel('시간 (s)'); ylabel('주파수 (Hz)');
ylim([0 fs/2]);

subplot(3,3,6);
diff_spec = 10*log10(abs(s_B_x).^2) - 10*log10(abs(s_A_x).^2);
imagesc(t_spec, f_spec, diff_spec);
axis xy; colorbar;
title('차이 (스트랩 B - A)');
xlabel('시간 (s)'); ylabel('주파수 (Hz)');
ylim([0 fs/2]);

% 스펙트로그램 - Y축
subplot(3,3,7);
[s_A_y, ~, ~] = spectrogram(acc_y_A, round(N/10), round(N/20), [], fs);
imagesc(t_spec, f_spec, 10*log10(abs(s_A_y).^2));
axis xy; colorbar;
title('스트랩 A - Y축 스펙트로그램');
xlabel('시간 (s)'); ylabel('주파수 (Hz)');
ylim([0 fs/2]);

subplot(3,3,8);
[s_B_y, ~, ~] = spectrogram(acc_y_B, round(N/10), round(N/20), [], fs);
imagesc(t_spec, f_spec, 10*log10(abs(s_B_y).^2));
axis xy; colorbar;
title('스트랩 B - Y축 스펙트로그램');
xlabel('시간 (s)'); ylabel('주파수 (Hz)');
ylim([0 fs/2]);

subplot(3,3,9);
diff_spec_y = 10*log10(abs(s_B_y).^2) - 10*log10(abs(s_A_y).^2);
imagesc(t_spec, f_spec, diff_spec_y);
axis xy; colorbar;
title('차이 (스트랩 B - A)');
xlabel('시간 (s)'); ylabel('주파수 (Hz)');
ylim([0 fs/2]);

sgtitle('통계적 비교 및 시간-주파수 분석', 'FontSize', 14, 'FontWeight', 'bold');

% Figure 3: 총 가속도 벡터 크기 분석
figure('Position', [100, 0, 1400, 600]);

% 총 가속도 벡터 크기
acc_magnitude_A = sqrt(acc_x_A.^2 + acc_y_A.^2 + acc_z_A.^2);
acc_magnitude_B = sqrt(acc_x_B.^2 + acc_y_B.^2 + acc_z_B.^2);

subplot(2,3,1);
plot(t(1:min(500, N)), acc_magnitude_A(1:min(500, N)), 'b-', 'LineWidth', 1.5); hold on;
plot(t(1:min(500, N)), acc_magnitude_B(1:min(500, N)), 'r--', 'LineWidth', 1.5);
title('총 가속도 벡터 크기 비교');
xlabel('시간 (s)'); ylabel('가속도 크기 (g)');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;

subplot(2,3,2);
histogram(acc_magnitude_A, 50, 'FaceColor', 'b', 'FaceAlpha', 0.5); hold on;
histogram(acc_magnitude_B, 50, 'FaceColor', 'r', 'FaceAlpha', 0.5);
title('가속도 크기 분포');
xlabel('가속도 크기 (g)'); ylabel('빈도');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;

subplot(2,3,3);
boxplot([acc_magnitude_A', acc_magnitude_B'], 'Labels', {'스트랩 A', '스트랩 B'});
title('가속도 크기 박스플롯');
ylabel('가속도 크기 (g)');
grid on;

% 고주파 성분 필터링 (고역 통과 필터)
% 5Hz 고역 통과 필터 설계
[b, a] = butter(4, 5/(fs/2), 'high');
acc_x_A_high = filter(b, a, acc_x_A);
acc_y_A_high = filter(b, a, acc_y_A);
acc_z_A_high = filter(b, a, acc_z_A);
acc_x_B_high = filter(b, a, acc_x_B);
acc_y_B_high = filter(b, a, acc_y_B);
acc_z_B_high = filter(b, a, acc_z_B);

acc_magnitude_A_high = sqrt(acc_x_A_high.^2 + acc_y_A_high.^2 + acc_z_A_high.^2);
acc_magnitude_B_high = sqrt(acc_x_B_high.^2 + acc_y_B_high.^2 + acc_z_B_high.^2);

subplot(2,3,4);
plot(t(1:min(500, N)), acc_magnitude_A_high(1:min(500, N)), 'b-', 'LineWidth', 1.5); hold on;
plot(t(1:min(500, N)), acc_magnitude_B_high(1:min(500, N)), 'r--', 'LineWidth', 1.5);
title('고주파 성분만 추출 (5Hz 이상)');
xlabel('시간 (s)'); ylabel('가속도 크기 (g)');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;

subplot(2,3,5);
histogram(acc_magnitude_A_high, 50, 'FaceColor', 'b', 'FaceAlpha', 0.5); hold on;
histogram(acc_magnitude_B_high, 50, 'FaceColor', 'r', 'FaceAlpha', 0.5);
title('고주파 성분 분포');
xlabel('가속도 크기 (g)'); ylabel('빈도');
legend('스트랩 A', '스트랩 B', 'Location', 'best');
grid on;

subplot(2,3,6);
boxplot([acc_magnitude_A_high', acc_magnitude_B_high'], 'Labels', {'스트랩 A', '스트랩 B'});
title('고주파 성분 박스플롯');
ylabel('가속도 크기 (g)');
grid on;

sgtitle('총 가속도 벡터 및 고주파 성분 분석', 'FontSize', 14, 'FontWeight', 'bold');

%% 6. 결과 요약 및 저장
fprintf('\n=== 분석 결과 요약 ===\n');
fprintf('스트랩 A (타이트): 미세 진동이 적음\n');
fprintf('스트랩 B (느슨): 미세 진동이 많음\n\n');

fprintf('주요 차이점:\n');
fprintf('1. RMS 값: 스트랩 B가 %.1f%% 높음\n', 100*(rms_total_B-rms_total_A)/rms_total_A);
fprintf('2. 표준편차: 스트랩 B가 %.1f%% 높음\n', 100*(std_total_B-std_total_A)/std_total_A);
fprintf('3. 고주파 파워 (5Hz 이상): 스트랩 B가 %.1f%% 높음\n', ...
        100*(high_freq_power_total_B-high_freq_power_total_A)/high_freq_power_total_A);

% 결과 저장
results.rms = struct('A', struct('x', rms_x_A, 'y', rms_y_A, 'z', rms_z_A, 'total', rms_total_A), ...
                     'B', struct('x', rms_x_B, 'y', rms_y_B, 'z', rms_z_B, 'total', rms_total_B));
results.std = struct('A', struct('x', std_x_A, 'y', std_y_A, 'z', std_z_A, 'total', std_total_A), ...
                     'B', struct('x', std_x_B, 'y', std_y_B, 'z', std_z_B, 'total', std_total_B));
results.high_freq_power = struct('A', struct('x', high_freq_power_x_A, 'y', high_freq_power_y_A, ...
                                              'z', high_freq_power_z_A, 'total', high_freq_power_total_A), ...
                                 'B', struct('x', high_freq_power_x_B, 'y', high_freq_power_y_B, ...
                                              'z', high_freq_power_z_B, 'total', high_freq_power_total_B));
results.fs = fs;
results.duration = duration;
results.freq = freq;
results.Pxx_A = struct('x', Pxx_x_A, 'y', Pxx_y_A, 'z', Pxx_z_A);
results.Pxx_B = struct('x', Pxx_x_B, 'y', Pxx_y_B, 'z', Pxx_z_B);

save('strap_acc_vibration_results.mat', 'results', 'acc_x_A', 'acc_y_A', 'acc_z_A', ...
     'acc_x_B', 'acc_y_B', 'acc_z_B', 't', 'freq');

fprintf('\n결과가 ''strap_acc_vibration_results.mat'' 파일에 저장되었습니다.\n');
fprintf('분석 완료!\n');
