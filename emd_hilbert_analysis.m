%% EMD 계열 방법을 이용한 25Hz 신호의 IMF-Hilbert 에너지 분석
% Author: MATLAB Signal Processing
% Date: 2024

clear all; close all; clc;

%% 1. 신호 생성 (25Hz 신호 포함)
fs = 1000;              % 샘플링 주파수 (Hz)
t = 0:1/fs:10-1/fs;     % 시간 벡터 (10초)
N = length(t);

% 복합 신호 생성 (25Hz 주성분 + 노이즈 + 다른 주파수 성분)
f1 = 25;                % 주요 주파수 25Hz
f2 = 5;                 % 저주파 성분
f3 = 50;                % 고주파 성분

% 시간에 따라 변하는 진폭을 가진 25Hz 신호
amp_modulation = 1 + 0.5*sin(2*pi*0.1*t);  % 0.1Hz로 진폭 변조
signal = amp_modulation .* sin(2*pi*f1*t) + ...
         0.5*sin(2*pi*f2*t) + ...
         0.3*sin(2*pi*f3*t) + ...
         0.1*randn(size(t));  % 백색 잡음

%% 2. 원본 신호 시각화
figure(1);
subplot(3,1,1);
plot(t(1:1000), signal(1:1000));
title('원본 신호 (첫 1초)');
xlabel('시간 (s)'); ylabel('진폭');
grid on;

subplot(3,1,2);
plot(t, signal);
title('전체 원본 신호');
xlabel('시간 (s)'); ylabel('진폭');
grid on;

% 주파수 스펙트럼
subplot(3,1,3);
[Pxx, f] = pwelch(signal, [], [], [], fs);
semilogy(f, Pxx);
title('원본 신호 파워 스펙트럼');
xlabel('주파수 (Hz)'); ylabel('파워');
grid on;
xlim([0 100]);

%% 3. EMD (Empirical Mode Decomposition) 분석
fprintf('EMD 분석 수행 중...\n');
try
    % MATLAB의 emd 함수 사용 (Signal Processing Toolbox 필요)
    [imf_emd, residual_emd] = emd(signal, 'MaxNumIMF', 8);
    
    % IMF 개수
    num_imf_emd = size(imf_emd, 2);
    fprintf('EMD: %d개의 IMF 추출됨\n', num_imf_emd);
    
    % EMD 결과 시각화
    figure(2);
    subplot(num_imf_emd+2, 1, 1);
    plot(t, signal);
    title('원본 신호');
    ylabel('진폭');
    
    for i = 1:num_imf_emd
        subplot(num_imf_emd+2, 1, i+1);
        plot(t, imf_emd(:,i));
        title(['IMF ', num2str(i)]);
        ylabel('진폭');
    end
    
    subplot(num_imf_emd+2, 1, num_imf_emd+2);
    plot(t, residual_emd);
    title('잔여 성분');
    xlabel('시간 (s)'); ylabel('진폭');
    
catch ME
    fprintf('EMD 오류: %s\n', ME.message);
    fprintf('사용자 정의 EMD 구현을 사용합니다.\n');
    
    % 간단한 EMD 구현
    [imf_emd, residual_emd] = simple_emd(signal);
    num_imf_emd = size(imf_emd, 2);
end

%% 4. EEMD (Ensemble Empirical Mode Decomposition) 분석
fprintf('EEMD 분석 수행 중...\n');

% EEMD 파라미터
num_ensembles = 100;    % 앙상블 개수
noise_std = 0.1;        % 노이즈 표준편차

% EEMD 구현
imf_eemd = zeros(N, 8);  % 최대 8개 IMF
residual_eemd = zeros(N, 1);

for ensemble = 1:num_ensembles
    % 백색 잡음 추가
    noisy_signal = signal + noise_std * randn(size(signal));
    
    try
        [temp_imf, temp_residual] = emd(noisy_signal, 'MaxNumIMF', 8);
    catch
        [temp_imf, temp_residual] = simple_emd(noisy_signal);
    end
    
    % 앙상블 평균
    num_temp_imf = size(temp_imf, 2);
    imf_eemd(:, 1:num_temp_imf) = imf_eemd(:, 1:num_temp_imf) + temp_imf;
    residual_eemd = residual_eemd + temp_residual;
    
    if mod(ensemble, 20) == 0
        fprintf('EEMD 진행률: %d/%d\n', ensemble, num_ensembles);
    end
end

% 평균 계산
imf_eemd = imf_eemd / num_ensembles;
residual_eemd = residual_eemd / num_ensembles;

% 유효한 IMF만 선택
valid_imf_idx = find(sum(abs(imf_eemd)) > 1e-6);
imf_eemd = imf_eemd(:, valid_imf_idx);
num_imf_eemd = length(valid_imf_idx);

fprintf('EEMD: %d개의 IMF 추출됨\n', num_imf_eemd);

%% 5. Hilbert 변환을 이용한 순간 주파수 및 에너지 분석
fprintf('Hilbert 변환 분석 수행 중...\n');

% EMD IMF에 대한 Hilbert 분석
hilbert_results_emd = cell(num_imf_emd, 1);
instantaneous_freq_emd = zeros(N, num_imf_emd);
instantaneous_amp_emd = zeros(N, num_imf_emd);
energy_emd = zeros(num_imf_emd, 1);

for i = 1:num_imf_emd
    % Hilbert 변환
    analytic_signal = hilbert(imf_emd(:,i));
    
    % 순간 진폭 및 위상
    instantaneous_amp_emd(:,i) = abs(analytic_signal);
    instantaneous_phase = unwrap(angle(analytic_signal));
    
    % 순간 주파수 (Hz)
    instantaneous_freq_emd(:,i) = fs/(2*pi) * diff([instantaneous_phase(1); instantaneous_phase]);
    
    % 에너지 계산
    energy_emd(i) = sum(instantaneous_amp_emd(:,i).^2) / N;
    
    hilbert_results_emd{i} = struct('amplitude', instantaneous_amp_emd(:,i), ...
                                   'frequency', instantaneous_freq_emd(:,i), ...
                                   'energy', energy_emd(i));
end

% EEMD IMF에 대한 Hilbert 분석
hilbert_results_eemd = cell(num_imf_eemd, 1);
instantaneous_freq_eemd = zeros(N, num_imf_eemd);
instantaneous_amp_eemd = zeros(N, num_imf_eemd);
energy_eemd = zeros(num_imf_eemd, 1);

for i = 1:num_imf_eemd
    % Hilbert 변환
    analytic_signal = hilbert(imf_eemd(:,i));
    
    % 순간 진폭 및 위상
    instantaneous_amp_eemd(:,i) = abs(analytic_signal);
    instantaneous_phase = unwrap(angle(analytic_signal));
    
    % 순간 주파수 (Hz)
    instantaneous_freq_eemd(:,i) = fs/(2*pi) * diff([instantaneous_phase(1); instantaneous_phase]);
    
    % 에너지 계산
    energy_eemd(i) = sum(instantaneous_amp_eemd(:,i).^2) / N;
    
    hilbert_results_eemd{i} = struct('amplitude', instantaneous_amp_eemd(:,i), ...
                                    'frequency', instantaneous_freq_eemd(:,i), ...
                                    'energy', energy_eemd(i));
end

%% 6. 25Hz 성분 식별 및 분석
fprintf('25Hz 성분 분석 중...\n');

% EMD에서 25Hz에 가장 가까운 IMF 찾기
target_freq = 25;
freq_tolerance = 5;  % ±5Hz 허용

% 각 IMF의 평균 주파수 계산
mean_freq_emd = zeros(num_imf_emd, 1);
for i = 1:num_imf_emd
    valid_freq_idx = (instantaneous_freq_emd(:,i) > 0) & (instantaneous_freq_emd(:,i) < fs/2);
    if sum(valid_freq_idx) > 0
        mean_freq_emd(i) = mean(instantaneous_freq_emd(valid_freq_idx,i));
    end
end

% 25Hz에 가장 가까운 IMF 찾기
[~, target_imf_emd] = min(abs(mean_freq_emd - target_freq));

% EEMD에서도 동일하게 수행
mean_freq_eemd = zeros(num_imf_eemd, 1);
for i = 1:num_imf_eemd
    valid_freq_idx = (instantaneous_freq_eemd(:,i) > 0) & (instantaneous_freq_eemd(:,i) < fs/2);
    if sum(valid_freq_idx) > 0
        mean_freq_eemd(i) = mean(instantaneous_freq_eemd(valid_freq_idx,i));
    end
end

[~, target_imf_eemd] = min(abs(mean_freq_eemd - target_freq));

%% 7. 결과 시각화
% Hilbert 스펙트럼 (EMD)
figure(3);
subplot(2,2,1);
for i = 1:num_imf_emd
    scatter(instantaneous_freq_emd(:,i), t, 10, instantaneous_amp_emd(:,i), 'filled');
    hold on;
end
colorbar;
title('EMD Hilbert 스펙트럼');
xlabel('순간 주파수 (Hz)'); ylabel('시간 (s)');
xlim([0 100]);

% Hilbert 스펙트럼 (EEMD)
subplot(2,2,2);
for i = 1:num_imf_eemd
    scatter(instantaneous_freq_eemd(:,i), t, 10, instantaneous_amp_eemd(:,i), 'filled');
    hold on;
end
colorbar;
title('EEMD Hilbert 스펙트럼');
xlabel('순간 주파수 (Hz)'); ylabel('시간 (s)');
xlim([0 100]);

% 에너지 분포 (EMD)
subplot(2,2,3);
bar(1:num_imf_emd, energy_emd);
title('EMD IMF 에너지 분포');
xlabel('IMF 번호'); ylabel('에너지');
grid on;

% 에너지 분포 (EEMD)
subplot(2,2,4);
bar(1:num_imf_eemd, energy_eemd);
title('EEMD IMF 에너지 분포');
xlabel('IMF 번호'); ylabel('에너지');
grid on;

% 25Hz 성분 상세 분석
figure(4);
subplot(2,2,1);
plot(t, imf_emd(:, target_imf_emd));
title(['EMD: 25Hz 성분 (IMF ', num2str(target_imf_emd), ')']);
xlabel('시간 (s)'); ylabel('진폭');
grid on;

subplot(2,2,2);
plot(t, instantaneous_freq_emd(:, target_imf_emd));
title('EMD: 25Hz 성분의 순간 주파수');
xlabel('시간 (s)'); ylabel('주파수 (Hz)');
grid on;
ylim([0 50]);

subplot(2,2,3);
plot(t, imf_eemd(:, target_imf_eemd));
title(['EEMD: 25Hz 성분 (IMF ', num2str(target_imf_eemd), ')']);
xlabel('시간 (s)'); ylabel('진폭');
grid on;

subplot(2,2,4);
plot(t, instantaneous_freq_eemd(:, target_imf_eemd));
title('EEMD: 25Hz 성분의 순간 주파수');
xlabel('시간 (s)'); ylabel('주파수 (Hz)');
grid on;
ylim([0 50]);

%% 8. 결과 요약 출력
fprintf('\n=== 분석 결과 요약 ===\n');
fprintf('원본 신호: 25Hz 주성분 + 5Hz, 50Hz 성분 + 노이즈\n');
fprintf('신호 길이: %.1f초, 샘플링 주파수: %dHz\n', t(end), fs);

fprintf('\nEMD 결과:\n');
fprintf('- 총 IMF 개수: %d\n', num_imf_emd);
fprintf('- 25Hz에 가장 가까운 IMF: %d번 (평균 주파수: %.2fHz)\n', target_imf_emd, mean_freq_emd(target_imf_emd));
fprintf('- 25Hz IMF 에너지: %.4f\n', energy_emd(target_imf_emd));

fprintf('\nEEMD 결과:\n');
fprintf('- 총 IMF 개수: %d\n', num_imf_eemd);
fprintf('- 25Hz에 가장 가까운 IMF: %d번 (평균 주파수: %.2fHz)\n', target_imf_eemd, mean_freq_eemd(target_imf_eemd));
fprintf('- 25Hz IMF 에너지: %.4f\n', energy_eemd(target_imf_eemd));

fprintf('\n각 IMF의 평균 주파수 (EMD):\n');
for i = 1:num_imf_emd
    fprintf('IMF %d: %.2f Hz (에너지: %.4f)\n', i, mean_freq_emd(i), energy_emd(i));
end

fprintf('\n각 IMF의 평균 주파수 (EEMD):\n');
for i = 1:num_imf_eemd
    fprintf('IMF %d: %.2f Hz (에너지: %.4f)\n', i, mean_freq_eemd(i), energy_eemd(i));
end

%% 9. 데이터 저장
save('emd_hilbert_results.mat', 'signal', 't', 'fs', ...
     'imf_emd', 'imf_eemd', 'residual_emd', 'residual_eemd', ...
     'hilbert_results_emd', 'hilbert_results_eemd', ...
     'energy_emd', 'energy_eemd', ...
     'instantaneous_freq_emd', 'instantaneous_freq_eemd', ...
     'instantaneous_amp_emd', 'instantaneous_amp_eemd');

fprintf('\n결과가 "emd_hilbert_results.mat" 파일로 저장되었습니다.\n');

%% 보조 함수: 간단한 EMD 구현
function [imf, residual] = simple_emd(signal)
    % 간단한 EMD 구현 (Signal Processing Toolbox가 없는 경우)
    
    imf = [];
    residual = signal(:);
    max_imf = 8;
    
    for imf_idx = 1:max_imf
        h = residual;
        
        % Sifting 과정
        for sift_iter = 1:10
            % 극값 찾기
            [max_vals, max_locs] = findpeaks(h);
            [min_vals, min_locs] = findpeaks(-h);
            min_vals = -min_vals;
            
            if length(max_locs) < 2 || length(min_locs) < 2
                break;
            end
            
            % 스플라인 보간으로 포락선 생성
            t_vec = 1:length(h);
            upper_env = interp1(max_locs, max_vals, t_vec, 'spline', 'extrap');
            lower_env = interp1(min_locs, min_vals, t_vec, 'spline', 'extrap');
            
            % 평균 제거
            mean_env = (upper_env + lower_env) / 2;
            new_h = h - mean_env(:);
            
            % 수렴 조건 확인
            if sum((h - new_h).^2) / sum(h.^2) < 0.01
                break;
            end
            
            h = new_h;
        end
        
        % IMF 저장
        imf = [imf, h];
        residual = residual - h;
        
        % 종료 조건
        if sum(abs(residual)) < 0.01 * sum(abs(signal))
            break;
        end
    end
end