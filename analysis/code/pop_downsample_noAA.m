function [EEG, command] = pop_downsample_noAA(EEG, new_srate)
% POP_DOWNSAMPLE_NOAA - Downsample EEG data without anti-aliasing filter.
%    Keeps event and urevent latencies consistent with EEGLAB's internal timing.
%
% Usage:
%   >> EEG = pop_downsample_noAA(EEG, new_srate);
%
% Inputs:
%   EEG        - input EEGLAB dataset
%   new_srate  - new sampling rate (Hz)
%
% Output:
%   EEG        - downsampled dataset (without anti-alias filtering)
%
% Notes:
%   - Intended for use when later filtering (e.g., 1–15 Hz) already
%     attenuates frequencies above the new Nyquist limit.
%   - Keeps event timing consistent so that pop_epoch works correctly.
%
% Author: Ders Dur + ChatGPT (2025), inspired by Arnaud Delorme's pop_resample

command = '';
if nargin < 2
    help pop_downsample_noAA;
    return;
end

if isempty(EEG(1).data)
    error('Cannot downsample empty dataset.');
end

% Handle multiple datasets
if length(EEG) > 1
    [EEG, command] = eeg_eval('pop_downsample_noAA', EEG, 'params', { new_srate });
    return;
end

old_srate = EEG.srate;
factor = old_srate / new_srate;

if abs(round(factor) - factor) > 1e-6
    error('Downsampling factor must be integer. Use standard pop_resample otherwise.');
end

factor = round(factor);
fprintf('Downsampling from %.2f Hz to %.2f Hz (factor = %d, NO ANTIALIAS FILTER)\n', old_srate, new_srate, factor);

% --- Downsample the data
EEG.data = downsample(EEG.data', factor)';  % Downsample along time dimension
EEG.pnts = size(EEG.data, 2);
EEG.srate = new_srate;
EEG.xmax  = EEG.xmin + (EEG.pnts - 1) / EEG.srate;
EEG.times = linspace(EEG.xmin * 1000, EEG.xmax * 1000, EEG.pnts);

% --- Adjust event latencies
if isfield(EEG, 'event') && ~isempty(EEG.event)
    fprintf('Adjusting event latencies...\n');
    for iEvt = 1:length(EEG.event)
        EEG.event(iEvt).latency = EEG.event(iEvt).latency / factor;
        if isfield(EEG.event(iEvt), 'duration') && ~isempty(EEG.event(iEvt).duration)
            EEG.event(iEvt).duration = EEG.event(iEvt).duration / factor;
        end
    end

    % Adjust urevent latencies if they exist
    if isfield(EEG, 'urevent') && isfield(EEG.urevent, 'latency')
        for iUrevt = 1:length(EEG.urevent)
            EEG.urevent(iUrevt).latency = EEG.urevent(iUrevt).latency / factor;
            if isfield(EEG.urevent(iUrevt), 'duration') && ~isempty(EEG.urevent(iUrevt).duration)
                EEG.urevent(iUrevt).duration = EEG.urevent(iUrevt).duration / factor;
            end
        end
    end

    EEG = eeg_checkset(EEG, 'eventconsistency');
end

EEG.icaact = [];
EEG.setname = sprintf('%s downsampled to %dHz (noAA)', EEG.setname, new_srate);
fprintf('Downsampling complete.\n');

command = sprintf('EEG = pop_downsample_noAA(EEG, %d);', new_srate);
end
