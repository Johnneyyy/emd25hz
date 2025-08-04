%% EMD 계열 방법들의 종합 비교 분석 (EMD, EEMD, FIF)
% 25Hz 신호에 대한 IMF-Hilbert 에너지 분석 종합 비교
% Author: MATLAB Signal Processing
% Date: 2024

clear all; close all; clc;

%% 1. 신호 생성 (25Hz 신호 포함)
fs = 1000;              % 샘플링 주파수 (Hz)
t = 0:1/fs:10-1/fs;     % 시간 벡터 (10초)
N = length(t);

% 복잡한 복합 신호 생성 (25Hz 주성분 + 다양한 특성)
f1 = 25;                % 주요 주파수 25Hz
f2 = 5;                 % 저주파 성분
f3 = 50;                % 고주파 성분
f4 = 75;                % 추가 고주파 성분

% 시간에 따라 변하는 특성을 가진 복합 신호
amp_modulation = 1 + 0.5*sin(2*pi*0.1*t);      % 진폭 변조
freq_modulation = 1 + 0.1*sin(2*pi*0.05*t);    % 주파수 변조
chirp_component = sin(2*pi*(10 + 30*t/max(t)).*t); % 처프 신호

signal = amp_modulation .* sin(2*pi*f1*freq_modulation.*t) + ...
         0.5*sin(2*pi*f2*t) + ...
         0.3*sin(2*pi*f3*t) + ...
         0.2*sin(2*pi*f4*t) + ...
         0.15*chirp_component + ...
         0.1*randn(size(t));  % 백색 잡음

fprintf('=== 25Hz 신호 EMD 계열 방법 종합 분석 ===\n');
fprintf('신호 특성: 25Hz 주성분 (진폭/주파수 변조) + 다중 주파수 + 처프 + 노이즈\n');
fprintf('신호 길이: %.1f초, 샘플링 주파수: %dHz\n\n', t(end), fs);

%% 2. 원본 신호 분석
figure(1);
subplot(2,2,1);
plot(t(1:2000), signal(1:2000));
title('원본 신호 (첫 2초)');
xlabel('시간 (s)'); ylabel('진폭');
grid on;

subplot(2,2,2);
[Pxx, f] = pwelch(signal, [], [], [], fs);
semilogy(f, Pxx);
title('원본 신호 파워 스펙트럼');
xlabel('주파수 (Hz)'); ylabel('파워');
grid on;
xlim([0 100]);

% 시간-주파수 분석 (스펙트로그램)
subplot(2,2,3);
spectrogram(signal, hamming(256), 128, 256, fs, 'yaxis');
title('원본 신호 스펙트로그램');
ylim([0 100]);

% 3D 시각화
subplot(2,2,4);
plot3(t(1:1000), real(hilbert(signal(1:1000))), imag(hilbert(signal(1:1000))));
title('원본 신호 3D 궤적 (Hilbert)');
xlabel('시간'); ylabel('실수부'); zlabel('허수부');
grid on;

%% 3. EMD 분석
fprintf('EMD 분석 수행 중...\n');
tic;
try
    [imf_emd, residual_emd] = emd(signal, 'MaxNumIMF', 8);
    num_imf_emd = size(imf_emd, 2);
catch ME
    fprintf('EMD 오류: %s\n', ME.message);
    [imf_emd, residual_emd] = simple_emd(signal);
    num_imf_emd = size(imf_emd, 2);
end
emd_time = toc;
fprintf('EMD 완료: %.2f초, %d개 IMF\n', emd_time, num_imf_emd);

%% 4. EEMD 분석
fprintf('EEMD 분석 수행 중...\n');
tic;
num_ensembles = 50;    % 계산 시간을 위해 줄임
noise_std = 0.1;
imf_eemd = zeros(N, 8);
residual_eemd = zeros(N, 1);

for ensemble = 1:num_ensembles
    noisy_signal = signal + noise_std * randn(size(signal));
    try
        [temp_imf, temp_residual] = emd(noisy_signal, 'MaxNumIMF', 8);
    catch
        [temp_imf, temp_residual] = simple_emd(noisy_signal);
    end
    
    num_temp_imf = size(temp_imf, 2);
    imf_eemd(:, 1:num_temp_imf) = imf_eemd(:, 1:num_temp_imf) + temp_imf;
    residual_eemd = residual_eemd + temp_residual;
    
    if mod(ensemble, 10) == 0
        fprintf('EEMD 진행률: %d/%d\n', ensemble, num_ensembles);
    end
end

imf_eemd = imf_eemd / num_ensembles;
residual_eemd = residual_eemd / num_ensembles;
valid_imf_idx = find(sum(abs(imf_eemd)) > 1e-6);
imf_eemd = imf_eemd(:, valid_imf_idx);
num_imf_eemd = length(valid_imf_idx);
eemd_time = toc;
fprintf('EEMD 완료: %.2f초, %d개 IMF\n', eemd_time, num_imf_eemd);

%% 5. FIF 분석
fprintf('FIF 분석 수행 중...\n');
tic;
options.delta = 0.001;
options.ExtPoints = 3;
options.NIMFs = 8;
options.MaxInner = 200;
options.MonotoneMaskLength = true;

[imf_fif_raw, logM] = FIF(signal, options);
num_imf_fif = size(imf_fif_raw, 1) - 1;
imf_fif = imf_fif_raw(1:num_imf_fif, :)';
residual_fif = imf_fif_raw(end, :)';
fif_time = toc;
fprintf('FIF 완료: %.2f초, %d개 IMF\n', fif_time, num_imf_fif);

%% 6. Hilbert 변환 분석 (모든 방법)
fprintf('Hilbert 변환 분석 수행 중...\n');

% EMD Hilbert 분석
[hilbert_emd, energy_emd, mean_freq_emd] = analyze_hilbert(imf_emd, fs);

% EEMD Hilbert 분석
[hilbert_eemd, energy_eemd, mean_freq_eemd] = analyze_hilbert(imf_eemd, fs);

% FIF Hilbert 분석
[hilbert_fif, energy_fif, mean_freq_fif] = analyze_hilbert(imf_fif, fs);

%% 7. 25Hz 성분 식별
target_freq = 25;
[~, target_imf_emd] = min(abs(mean_freq_emd - target_freq));
[~, target_imf_eemd] = min(abs(mean_freq_eemd - target_freq));
[~, target_imf_fif] = min(abs(mean_freq_fif - target_freq));

%% 8. 성능 메트릭 계산
% 신호 재구성 오차
recon_error_emd = norm(signal(:) - sum(imf_emd, 2) - residual_emd) / norm(signal);
recon_error_eemd = norm(signal(:) - sum(imf_eemd, 2) - residual_eemd) / norm(signal);
recon_error_fif = norm(signal(:) - sum(imf_fif, 2) - residual_fif) / norm(signal);

% 모드 혼합 지수 (Mode Mixing Index)
mmi_emd = calculate_mode_mixing(imf_emd, fs);
mmi_eemd = calculate_mode_mixing(imf_eemd, fs);
mmi_fif = calculate_mode_mixing(imf_fif, fs);

%% 9. 종합 결과 시각화
% IMF 비교
figure(2);
max_imf = max([num_imf_emd, num_imf_eemd, num_imf_fif]);
for i = 1:min(6, max_imf)
    subplot(6, 3, (i-1)*3 + 1);
    if i <= num_imf_emd
        plot(t, imf_emd(:,i));
        title(['EMD IMF ', num2str(i)]);
    end
    ylabel('진폭');
    grid on;
    
    subplot(6, 3, (i-1)*3 + 2);
    if i <= num_imf_eemd
        plot(t, imf_eemd(:,i));
        title(['EEMD IMF ', num2str(i)]);
    end
    grid on;
    
    subplot(6, 3, (i-1)*3 + 3);
    if i <= num_imf_fif
        plot(t, imf_fif(:,i));
        title(['FIF IMF ', num2str(i)]);
    end
    grid on;
end

% 에너지 분포 비교
figure(3);
subplot(2,3,1);
bar(1:num_imf_emd, energy_emd);
title('EMD 에너지 분포');
xlabel('IMF'); ylabel('에너지');
grid on;

subplot(2,3,2);
bar(1:num_imf_eemd, energy_eemd);
title('EEMD 에너지 분포');
xlabel('IMF'); ylabel('에너지');
grid on;

subplot(2,3,3);
bar(1:num_imf_fif, energy_fif);
title('FIF 에너지 분포');
xlabel('IMF'); ylabel('에너지');
grid on;

% 25Hz 성분 비교
subplot(2,3,4);
plot(t, imf_emd(:, target_imf_emd));
title(['EMD 25Hz 성분 (IMF ', num2str(target_imf_emd), ')']);
xlabel('시간 (s)'); ylabel('진폭');
grid on;

subplot(2,3,5);
plot(t, imf_eemd(:, target_imf_eemd));
title(['EEMD 25Hz 성분 (IMF ', num2str(target_imf_eemd), ')']);
xlabel('시간 (s)'); ylabel('진폭');
grid on;

subplot(2,3,6);
plot(t, imf_fif(:, target_imf_fif));
title(['FIF 25Hz 성분 (IMF ', num2str(target_imf_fif), ')']);
xlabel('시간 (s)'); ylabel('진폭');
grid on;

% Hilbert 스펙트럼 비교
figure(4);
subplot(1,3,1);
plot_hilbert_spectrum(hilbert_emd.freq, t, hilbert_emd.amp, 'EMD Hilbert 스펙트럼');

subplot(1,3,2);
plot_hilbert_spectrum(hilbert_eemd.freq, t, hilbert_eemd.amp, 'EEMD Hilbert 스펙트럼');

subplot(1,3,3);
plot_hilbert_spectrum(hilbert_fif.freq, t, hilbert_fif.amp, 'FIF Hilbert 스펙트럼');

%% 10. 정량적 성능 비교
figure(5);
methods = {'EMD', 'EEMD', 'FIF'};
computation_times = [emd_time, eemd_time, fif_time];
reconstruction_errors = [recon_error_emd, recon_error_eemd, recon_error_fif];
mode_mixing_indices = [mmi_emd, mmi_eemd, mmi_fif];

subplot(2,2,1);
bar(computation_times);
set(gca, 'XTickLabel', methods);
title('계산 시간 비교');
ylabel('시간 (초)');
grid on;

subplot(2,2,2);
bar(reconstruction_errors);
set(gca, 'XTickLabel', methods);
title('재구성 오차 비교');
ylabel('정규화된 오차');
grid on;

subplot(2,2,3);
bar(mode_mixing_indices);
set(gca, 'XTickLabel', methods);
title('모드 혼합 지수 비교');
ylabel('MMI');
grid on;

% 25Hz 성분 에너지 비교
target_energies = [energy_emd(target_imf_emd), energy_eemd(target_imf_eemd), energy_fif(target_imf_fif)];
subplot(2,2,4);
bar(target_energies);
set(gca, 'XTickLabel', methods);
title('25Hz 성분 에너지 비교');
ylabel('에너지');
grid on;

%% 11. 결과 요약 출력
fprintf('\n=== 종합 분석 결과 요약 ===\n');
fprintf('방법\t\tIMF수\t계산시간(s)\t재구성오차\tMMI\t\t25Hz위치\t25Hz에너지\n');
fprintf('EMD\t\t%d\t%.2f\t\t%.6f\t%.4f\t\t%d\t\t%.4f\n', ...
    num_imf_emd, emd_time, recon_error_emd, mmi_emd, target_imf_emd, energy_emd(target_imf_emd));
fprintf('EEMD\t\t%d\t%.2f\t\t%.6f\t%.4f\t\t%d\t\t%.4f\n', ...
    num_imf_eemd, eemd_time, recon_error_eemd, mmi_eemd, target_imf_eemd, energy_eemd(target_imf_eemd));
fprintf('FIF\t\t%d\t%.2f\t\t%.6f\t%.4f\t\t%d\t\t%.4f\n', ...
    num_imf_fif, fif_time, recon_error_fif, mmi_fif, target_imf_fif, energy_fif(target_imf_fif));

fprintf('\n=== 25Hz 성분 주파수 정확도 ===\n');
fprintf('EMD:  %.2f Hz (오차: %.2f Hz)\n', mean_freq_emd(target_imf_emd), abs(mean_freq_emd(target_imf_emd) - 25));
fprintf('EEMD: %.2f Hz (오차: %.2f Hz)\n', mean_freq_eemd(target_imf_eemd), abs(mean_freq_eemd(target_imf_eemd) - 25));
fprintf('FIF:  %.2f Hz (오차: %.2f Hz)\n', mean_freq_fif(target_imf_fif), abs(mean_freq_fif(target_imf_fif) - 25));

%% 12. 데이터 저장
save('comprehensive_emd_comparison_results.mat', ...
     'signal', 't', 'fs', ...
     'imf_emd', 'imf_eemd', 'imf_fif', ...
     'residual_emd', 'residual_eemd', 'residual_fif', ...
     'hilbert_emd', 'hilbert_eemd', 'hilbert_fif', ...
     'energy_emd', 'energy_eemd', 'energy_fif', ...
     'mean_freq_emd', 'mean_freq_eemd', 'mean_freq_fif', ...
     'computation_times', 'reconstruction_errors', 'mode_mixing_indices');

fprintf('\n모든 결과가 "comprehensive_emd_comparison_results.mat" 파일로 저장되었습니다.\n');

%% 보조 함수들
function [hilbert_results, energies, mean_freqs] = analyze_hilbert(imfs, fs)
    N = size(imfs, 1);
    num_imfs = size(imfs, 2);
    
    instantaneous_freq = zeros(N, num_imfs);
    instantaneous_amp = zeros(N, num_imfs);
    energies = zeros(num_imfs, 1);
    mean_freqs = zeros(num_imfs, 1);
    
    for i = 1:num_imfs
        analytic_signal = hilbert(imfs(:,i));
        instantaneous_amp(:,i) = abs(analytic_signal);
        instantaneous_phase = unwrap(angle(analytic_signal));
        instantaneous_freq(:,i) = fs/(2*pi) * diff([instantaneous_phase(1); instantaneous_phase]);
        
        energies(i) = sum(instantaneous_amp(:,i).^2) / N;
        
        valid_freq_idx = (instantaneous_freq(:,i) > 0) & (instantaneous_freq(:,i) < fs/2);
        if sum(valid_freq_idx) > 0
            mean_freqs(i) = mean(instantaneous_freq(valid_freq_idx,i));
        end
    end
    
    hilbert_results = struct('freq', instantaneous_freq, 'amp', instantaneous_amp);
end

function mmi = calculate_mode_mixing(imfs, fs)
    % 간단한 모드 혼합 지수 계산
    num_imfs = size(imfs, 2);
    freq_overlaps = 0;
    
    for i = 1:num_imfs-1
        for j = i+1:num_imfs
            % 각 IMF의 주파수 범위 계산
            [~, f1] = pwelch(imfs(:,i), [], [], [], fs);
            [Pxx1, ~] = pwelch(imfs(:,i), [], [], [], fs);
            [~, f2] = pwelch(imfs(:,j), [], [], [], fs);
            [Pxx2, ~] = pwelch(imfs(:,j), [], [], [], fs);
            
            % 주파수 겹침 계산
            overlap = sum(min(Pxx1, Pxx2)) / (sum(Pxx1) + sum(Pxx2));
            freq_overlaps = freq_overlaps + overlap;
        end
    end
    
    mmi = freq_overlaps / (num_imfs * (num_imfs - 1) / 2);
end

function plot_hilbert_spectrum(freq_matrix, time_vector, amp_matrix, title_str)
    num_imfs = size(freq_matrix, 2);
    for i = 1:num_imfs
        scatter(freq_matrix(:,i), time_vector, 10, amp_matrix(:,i), 'filled');
        hold on;
    end
    colorbar;
    title(title_str);
    xlabel('순간 주파수 (Hz)');
    ylabel('시간 (s)');
    xlim([0 100]);
end

function [imf, residual] = simple_emd(signal)
    % 간단한 EMD 구현
    imf = [];
    residual = signal(:);
    max_imf = 8;
    
    for imf_idx = 1:max_imf
        h = residual;
        
        for sift_iter = 1:10
            [max_vals, max_locs] = findpeaks(h);
            [min_vals, min_locs] = findpeaks(-h);
            min_vals = -min_vals;
            
            if length(max_locs) < 2 || length(min_locs) < 2
                break;
            end
            
            t_vec = 1:length(h);
            upper_env = interp1(max_locs, max_vals, t_vec, 'spline', 'extrap');
            lower_env = interp1(min_locs, min_vals, t_vec, 'spline', 'extrap');
            
            mean_env = (upper_env + lower_env) / 2;
            new_h = h - mean_env(:);
            
            if sum((h - new_h).^2) / sum(h.^2) < 0.01
                break;
            end
            
            h = new_h;
        end
        
        imf = [imf, h];
        residual = residual - h;
        
        if sum(abs(residual)) < 0.01 * sum(abs(signal))
            break;
        end
    end
end

function [IMF, logM] = FIF(f, options)
    % FIF 구현 (이전과 동일)
    if nargin < 2
        options = struct();
    end
    
    if ~isfield(options, 'delta'), options.delta = 0.001; end
    if ~isfield(options, 'ExtPoints'), options.ExtPoints = 3; end
    if ~isfield(options, 'NIMFs'), options.NIMFs = 8; end
    if ~isfield(options, 'MaxInner'), options.MaxInner = 200; end
    if ~isfield(options, 'MonotoneMaskLength'), options.MonotoneMaskLength = true; end
    
    f = f(:)';
    N = length(f);
    IMF = [];
    logM = [];
    
    MM = 2:N/5;
    if options.MonotoneMaskLength
        MM = unique(round(logspace(log10(2), log10(N/5), options.NIMFs)));
    end
    
    h = f;
    
    for IMFIndex = 1:options.NIMFs
        if IMFIndex <= length(MM)
            maskLength = MM(IMFIndex);
        else
            maskLength = MM(end);
        end
        
        mask = ones(1, maskLength) / maskLength;
        
        for inner = 1:options.MaxInner
            h_old = h;
            
            h_ext = extend_signal(h, options.ExtPoints);
            h_filtered = conv(h_ext, mask, 'same');
            h_filtered = h_filtered((options.ExtPoints+1):(end-options.ExtPoints));
            
            h = h - h_filtered;
            
            if norm(h - h_old) / norm(h_old) < options.delta
                break;
            end
        end
        
        IMF = [IMF; h];
        
        f = f - h;
        h = f;
        
        if max(abs(f)) < options.delta * max(abs(IMF(1,:)))
            break;
        end
        
        logM = [logM; inner, norm(h)];
    end
    
    IMF = [IMF; f];
end

function extended_signal = extend_signal(signal, ext_points)
    N = length(signal);
    left_ext = signal(ext_points:-1:1);
    right_ext = signal(N:-1:(N-ext_points+1));
    extended_signal = [left_ext, signal, right_ext];
end