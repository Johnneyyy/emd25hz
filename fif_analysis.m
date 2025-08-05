%% FIF (Fast Iterative Filtering) 기반 25Hz 신호 분석
% FIF는 EMD의 개선된 버전으로 더 빠르고 안정적인 분해를 제공합니다.
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
f4 = 75;                % 추가 고주파 성분

% 비선형 및 비정상 특성을 가진 25Hz 신호
amp_modulation = 1 + 0.5*sin(2*pi*0.1*t);  % 0.1Hz로 진폭 변조
freq_modulation = 1 + 0.1*sin(2*pi*0.05*t); % 주파수 변조
signal = amp_modulation .* sin(2*pi*f1*freq_modulation.*t) + ...
         0.5*sin(2*pi*f2*t) + ...
         0.3*sin(2*pi*f3*t) + ...
         0.2*sin(2*pi*f4*t) + ...
         0.1*randn(size(t));  % 백색 잡음

%% 2. FIF (Fast Iterative Filtering) 구현
fprintf('FIF 분석 수행 중...\n');

% FIF 파라미터
options.delta = 0.001;          % 수렴 임계값
options.ExtPoints = 3;          % 경계 확장 점수
options.NIMFs = 8;              % 최대 IMF 개수
options.MaxInner = 200;         % 최대 내부 반복
options.MonotoneMaskLength = true;

% FIF 분해 수행
[imf_fif, logM] = FIF(signal, options);
num_imf_fif = size(imf_fif, 1) - 1;  % 마지막은 residual

% IMF와 residual 분리
imfs_fif = imf_fif(1:num_imf_fif, :)';
residual_fif = imf_fif(end, :)';

fprintf('FIF: %d개의 IMF 추출됨\n', num_imf_fif);

%% 3. FIF 결과 시각화
figure(1);
subplot(num_imf_fif+2, 1, 1);
plot(t, signal);
title('원본 신호');
ylabel('진폭');
grid on;

for i = 1:num_imf_fif
    subplot(num_imf_fif+2, 1, i+1);
    plot(t, imfs_fif(:,i));
    title(['FIF IMF ', num2str(i)]);
    ylabel('진폭');
    grid on;
end

subplot(num_imf_fif+2, 1, num_imf_fif+2);
plot(t, residual_fif);
title('FIF 잔여 성분');
xlabel('시간 (s)'); ylabel('진폭');
grid on;

%% 4. FIF IMF에 대한 Hilbert 변환 분석
fprintf('FIF Hilbert 변환 분석 수행 중...\n');

hilbert_results_fif = cell(num_imf_fif, 1);
instantaneous_freq_fif = zeros(N, num_imf_fif);
instantaneous_amp_fif = zeros(N, num_imf_fif);
energy_fif = zeros(num_imf_fif, 1);

for i = 1:num_imf_fif
    % Hilbert 변환
    analytic_signal = hilbert(imfs_fif(:,i));
    
    % 순간 진폭 및 위상
    instantaneous_amp_fif(:,i) = abs(analytic_signal);
    instantaneous_phase = unwrap(angle(analytic_signal));
    
    % 순간 주파수 (Hz)
    instantaneous_freq_fif(:,i) = fs/(2*pi) * diff([instantaneous_phase(1); instantaneous_phase]);
    
    % 에너지 계산
    energy_fif(i) = sum(instantaneous_amp_fif(:,i).^2) / N;
    
    hilbert_results_fif{i} = struct('amplitude', instantaneous_amp_fif(:,i), ...
                                   'frequency', instantaneous_freq_fif(:,i), ...
                                   'energy', energy_fif(i));
end

%% 5. 25Hz 성분 식별
fprintf('FIF 25Hz 성분 분석 중...\n');

target_freq = 25;

% 각 IMF의 평균 주파수 계산
mean_freq_fif = zeros(num_imf_fif, 1);
for i = 1:num_imf_fif
    valid_freq_idx = (instantaneous_freq_fif(:,i) > 0) & (instantaneous_freq_fif(:,i) < fs/2);
    if sum(valid_freq_idx) > 0
        mean_freq_fif(i) = mean(instantaneous_freq_fif(valid_freq_idx,i));
    end
end

% 25Hz에 가장 가까운 IMF 찾기
[~, target_imf_fif] = min(abs(mean_freq_fif - target_freq));

%% 6. 결과 시각화 및 비교
% Hilbert 스펙트럼 (FIF)
figure(2);
subplot(2,2,1);
for i = 1:num_imf_fif
    scatter(instantaneous_freq_fif(:,i), t, 10, instantaneous_amp_fif(:,i), 'filled');
    hold on;
end
colorbar;
title('FIF Hilbert 스펙트럼');
xlabel('순간 주파수 (Hz)'); ylabel('시간 (s)');
xlim([0 100]);

% 에너지 분포 (FIF)
subplot(2,2,2);
bar(1:num_imf_fif, energy_fif);
title('FIF IMF 에너지 분포');
xlabel('IMF 번호'); ylabel('에너지');
grid on;

% 25Hz 성분 (FIF)
subplot(2,2,3);
plot(t, imfs_fif(:, target_imf_fif));
title(['FIF: 25Hz 성분 (IMF ', num2str(target_imf_fif), ')']);
xlabel('시간 (s)'); ylabel('진폭');
grid on;

% 25Hz 성분의 순간 주파수 (FIF)
subplot(2,2,4);
plot(t, instantaneous_freq_fif(:, target_imf_fif));
title('FIF: 25Hz 성분의 순간 주파수');
xlabel('시간 (s)'); ylabel('주파수 (Hz)');
grid on;
ylim([0 50]);

%% 7. 주파수 도메인 분석
figure(3);
for i = 1:min(4, num_imf_fif)
    subplot(2,2,i);
    [Pxx, f] = pwelch(imfs_fif(:,i), [], [], [], fs);
    semilogy(f, Pxx);
    title(['FIF IMF ', num2str(i), ' 파워 스펙트럼']);
    xlabel('주파수 (Hz)'); ylabel('파워');
    grid on;
    xlim([0 100]);
end

%% 8. 결과 요약 출력
fprintf('\n=== FIF 분석 결과 요약 ===\n');
fprintf('원본 신호: 25Hz 주성분 (진폭/주파수 변조) + 다중 주파수 성분 + 노이즈\n');
fprintf('신호 길이: %.1f초, 샘플링 주파수: %dHz\n', t(end), fs);

fprintf('\nFIF 결과:\n');
fprintf('- 총 IMF 개수: %d\n', num_imf_fif);
fprintf('- 25Hz에 가장 가까운 IMF: %d번 (평균 주파수: %.2fHz)\n', target_imf_fif, mean_freq_fif(target_imf_fif));
fprintf('- 25Hz IMF 에너지: %.4f\n', energy_fif(target_imf_fif));

fprintf('\n각 IMF의 평균 주파수 (FIF):\n');
for i = 1:num_imf_fif
    fprintf('IMF %d: %.2f Hz (에너지: %.4f)\n', i, mean_freq_fif(i), energy_fif(i));
end

%% 9. 데이터 저장
save('fif_hilbert_results.mat', 'signal', 't', 'fs', ...
     'imfs_fif', 'residual_fif', 'hilbert_results_fif', ...
     'energy_fif', 'instantaneous_freq_fif', 'instantaneous_amp_fif', ...
     'mean_freq_fif', 'target_imf_fif');

fprintf('\nFIF 결과가 "fif_hilbert_results.mat" 파일로 저장되었습니다.\n');

%% FIF (Fast Iterative Filtering) 구현 함수
function [IMF, logM] = FIF(f, options)
    % FIF - Fast Iterative Filtering
    % 입력:
    %   f: 입력 신호 (벡터)
    %   options: FIF 옵션 구조체
    % 출력:
    %   IMF: 분해된 IMF들 (각 행이 하나의 IMF)
    %   logM: 로그 정보
    
    if nargin < 2
        options = struct();
    end
    
    % 기본 옵션 설정
    if ~isfield(options, 'delta'), options.delta = 0.001; end
    if ~isfield(options, 'ExtPoints'), options.ExtPoints = 3; end
    if ~isfield(options, 'NIMFs'), options.NIMFs = 8; end
    if ~isfield(options, 'MaxInner'), options.MaxInner = 200; end
    if ~isfield(options, 'MonotoneMaskLength'), options.MonotoneMaskLength = true; end
    
    f = f(:)';  % 행 벡터로 변환
    N = length(f);
    IMF = [];
    logM = [];
    
    % 마스크 길이 계산
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
        
        % 마스크 생성
        mask = ones(1, maskLength) / maskLength;
        
        % 반복 필터링
        for inner = 1:options.MaxInner
            h_old = h;
            
            % 확장된 신호에 필터 적용
            h_ext = extend_signal(h, options.ExtPoints);
            h_filtered = conv(h_ext, mask, 'same');
            h_filtered = h_filtered((options.ExtPoints+1):(end-options.ExtPoints));
            
            % IMF 업데이트
            h = h - h_filtered;
            
            % 수렴 조건 확인
            if norm(h - h_old) / norm(h_old) < options.delta
                break;
            end
        end
        
        % IMF 저장
        IMF = [IMF; h];
        
        % 잔여 신호 업데이트
        f = f - h;
        h = f;
        
        % 종료 조건 확인
        if max(abs(f)) < options.delta * max(abs(IMF(1,:)))
            break;
        end
        
        logM = [logM; inner, norm(h)];
    end
    
    % 잔여 성분 추가
    IMF = [IMF; f];
end

function extended_signal = extend_signal(signal, ext_points)
    % 신호 양 끝을 확장하는 함수
    N = length(signal);
    
    % 좌측 확장 (미러링)
    left_ext = signal(ext_points:-1:1);
    
    % 우측 확장 (미러링)
    right_ext = signal(N:-1:(N-ext_points+1));
    
    extended_signal = [left_ext, signal, right_ext];
end