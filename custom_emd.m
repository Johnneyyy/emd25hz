function [imfs, residual] = custom_emd(signal, max_imfs)
% CUSTOM_EMD - Simple implementation of Empirical Mode Decomposition
% 
% Inputs:
%   signal - Input signal (column vector)
%   max_imfs - Maximum number of IMFs to extract
%
% Outputs:
%   imfs - Matrix where each column is an IMF
%   residual - The final residual after extracting all IMFs

    if nargin < 2
        max_imfs = 10;
    end
    
    % Ensure signal is a column vector
    signal = signal(:);
    n = length(signal);
    
    % Initialize
    imfs = [];
    residual = signal;
    imf_count = 0;
    
    % EMD main loop
    while imf_count < max_imfs
        h = residual;
        SD = 1;
        num_sifts = 0;
        max_sifts = 100;
        
        % Sifting process
        while SD > 0.2 && num_sifts < max_sifts
            % Find local maxima and minima
            [max_vals, max_locs] = findpeaks(h);
            [min_vals, min_locs] = findpeaks(-h);
            min_vals = -min_vals;
            
            % Check if we have enough extrema
            if length(max_locs) < 2 || length(min_locs) < 2
                break;
            end
            
            % Handle boundary conditions
            % Extend extrema by mirroring at boundaries
            if max_locs(1) > min_locs(1)
                max_locs = [1; max_locs];
                max_vals = [h(1); max_vals];
            end
            if min_locs(1) > max_locs(1)
                min_locs = [1; min_locs];
                min_vals = [h(1); min_vals];
            end
            if max_locs(end) < min_locs(end)
                max_locs = [max_locs; n];
                max_vals = [max_vals; h(n)];
            end
            if min_locs(end) < max_locs(end)
                min_locs = [min_locs; n];
                min_vals = [min_vals; h(n)];
            end
            
            % Create upper and lower envelopes using spline interpolation
            try
                upper_env = interp1(max_locs, max_vals, 1:n, 'spline', 'extrap')';
                lower_env = interp1(min_locs, min_vals, 1:n, 'spline', 'extrap')';
            catch
                % If spline fails, use linear interpolation
                upper_env = interp1(max_locs, max_vals, 1:n, 'linear', 'extrap')';
                lower_env = interp1(min_locs, min_vals, 1:n, 'linear', 'extrap')';
            end
            
            % Calculate mean of envelopes
            mean_env = (upper_env + lower_env) / 2;
            
            % Update h
            h_prev = h;
            h = h - mean_env;
            
            % Calculate stopping criterion (Cauchy convergence)
            SD = sum((h_prev - h).^2) / sum(h_prev.^2);
            num_sifts = num_sifts + 1;
        end
        
        % Check if h is an IMF (has at least 2 extrema)
        [~, max_locs] = findpeaks(h);
        [~, min_locs] = findpeaks(-h);
        
        if length(max_locs) + length(min_locs) < 3
            % No more IMFs can be extracted
            break;
        end
        
        % Store the IMF
        imf_count = imf_count + 1;
        if imf_count == 1
            imfs = h;
        else
            imfs(:, imf_count) = h;
        end
        
        % Update residual
        residual = residual - h;
        
        % Check if residual is monotonic
        if is_monotonic(residual)
            break;
        end
    end
    
    % If no IMFs were extracted, return the signal as a single IMF
    if isempty(imfs)
        imfs = signal;
        residual = zeros(size(signal));
    end
end

function result = is_monotonic(signal)
% Check if signal is monotonic
    d = diff(signal);
    result = all(d >= 0) || all(d <= 0);
end