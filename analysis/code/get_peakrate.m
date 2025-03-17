%TODO: FIGURE OUT IF STILL USEFUL - MOVE TO STIMULI OR MAKE SUBFUNCTION OF
%TRF_ANALYSIS IF SO
function [peakrate, smooth_env, denv]= get_peakrate(x, fs)
    % [peakrate, smooth_env, denv]= get_peakrate(x,fs)
    % assumes x is [time x channels] waveform vector with identical
    % waveforms on each channel (so it can throw one out), then calculates
    % broadband envelope, smooths it using lowpass butterworth 0 phase,
    % then takes derivative and rectifies.
    % peakrate is a vector with indices corresponding to derivative peaks

    % set threshold for peaks to ignore based on smoothed envelope value
    env_thresh = 0.003;
    % envelope lowpass smoothing factor
    
    x = x(:,1); % both channels are the same
    smooth_env = get_env(x, fs);
    % note get_env alread smooths below 10 Hz
    
    % differentiate
    denv = zeros(length(smooth_env), 1);
    denv(2:end,:) = diff(smooth_env);
    % rectify
    denv(...
        denv<0) = 0; 
    %note: thresholding does make significant difference
    % find peaks in env rate
    [~, peak_locs] = findpeaks(denv);
    % remove peaks where envelope is tiny
    mask = zeros(size(x));
    mask(peak_locs) = 1;
    mask(smooth_env<env_thresh) = 0;
    peakrate = find(mask);
end