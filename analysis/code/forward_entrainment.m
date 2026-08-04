
%% FORWARD ENTRAINMENT ANALYSIS
% -------------------------------------------------------------------------
% Searches for signatures of forward entrainment (persistent neural
% activity during silences) using two complementary analyses:
% 
% 1. Channel-wise ERPs time-locked to the onset of detected silences
% 2. Dual-feature TRFs (envelope + binary silece-onset regressor) fir with
% ridge parameter (lambda) chosen via nested cross-validation, then refit
% on the full dataset at the optimal lambda
% -------------------------------------------------------------------------

clear; clc; close all;
%% ------------------------------------------------------------------------
% USER CONFIG
% -------------------------------------------------------------------------


cfg=[];
cfg.fs=128; % Hz
cfg.silenceThresh= 0.05; % envelope threshold below
                            % which samples count as silence.

cfg.minSilenceDur=0.3; % in (s) NOTE: Saberi & Hickok reported fwd entrainment
% falls off outside of 3-5 Hz, so minimum duration for a single cycle
% should be around .33 s
%TODO: is there a principled way to set this threshold?

cfg.erpWindow=[-.1, .6]; % [pre, post] (s) epoch window around silence onset
% NOTE: silent periods shorter than duration implied by window are
% excluded from the ERP analysis


cfg.trfLags=[-100,500]; %[tmin, tmax] (ms)
cfg.lambdaGrid = 10.^(-2:1:6); %

cfg.roiElectrodes=[54 55 56 61 62 63 106 107 108 115 116 117]; % electrodes 
% to average for ROI plots

cfg.baselineWinERP=[-0.1 0]; % window to define GFP threshold, ERP
cfg.baselineWinTRF=[-100 0]; % TRF lags for GFP threshold (ms)


%% ------------------------------------------------------------------------
% LOAD DATA
% -------------------------------------------------------------------------
subj=24;
trf_analysis_params
% clear script_config
pp_checkpoint_=load_checkpoint(preprocess_config);
EEG=pp_checkpoint_.preprocessed_eeg;
envCell=load_stim_cell(trf_config.paths.envelopesFile, EEG.cond, EEG.trials);

clear do_nulltest train_params show_tuning_curves pp_checkpoint_

% define chanlocs by running trf_analysis
load(loc_file)

%% ------------------------------------------------------------------------
% SILENCE DETECTION
% -------------------------------------------------------------------------
nStim=numel(envCell);
silCell=cell(nStim,1);

for ii=1:nStim
    silCell{ii}=detectSilences(envCell{ii}',cfg.fs,cfg.silenceThresh, ...
        cfg.minSilenceDur);
end

fprintf('Detected %d total silent periods across %d stimuli.\n', ...
    sum(cellfun(@numel, silCell)), nStim);

%% ------------------------------------------------------------------------
% ERP calculation (time-locked to silent onset)
% -------------------------------------------------------------------------
[erpData,erpTimes]=computeERPs(EEG.resp,silCell,cfg.fs,cfg.erpWindow);
% erpData: nChn x nTimes 
%% ------------------------------------------------------------------------
% Local Functions
% -------------------------------------------------------------------------

function silenceOffsets=detectSilences(env,fs,thresh,minDur)
% Find onset sample indices of silence periods
% env : 1 x T envelope vector
% fs : sampling rate (Hz)
% thresh: envelope threshold below which a sample is "silent"
% minDur : minimum duration (s) for a silent period to be kept
%
% silenceOnsets : vector of sample indices marking the FIRST sample of each
% qualifying silent interval
    belowThresh=env<thresh; % logical, 1 x T
    % identify contiguous runs of belowThresh == true, and keep only the
    % runs whose length in samples is >= mindDur*fs
    d=diff([0 belowThresh 0]); % pad with virtual zeros to accurately 
                                % detect & index transitions at edges
    runStarts=find(d==1); runEnds=find(d==-1);
    silenceLengths=runEnds-runStarts+1; % samples
    keep=silenceLengths>=minDur*fs;
    % Take the first sample of each surviving run as its onset
    %NOTE: MIGHT WANT TO EXCLUDE THE START OF A TRIAL??????????
    silenceOffsets=runStarts(keep);
end

function [erpData,erpTimes]=computeERPs(eegCell,silenceCell,fs,epochWin)
% Epoch EEG around silence onsets, pool across stimuli, average across
% epochs
%
% eegCell{i} : T_i x nChns EEG matrix for stimulus i
% silenceCell{i} : sample indices of silence onsets for stimulus i
% fs: sampling rate (Hz)
% epochWin : [preSec postSec] window around onsets
%
% erpData : nChns x nTimes (baseline-corrected, pooled across all stimuli
% and all onsets, averaged across epochs)
%
% erpTimes : 1 x nTimes time axis in seconds relative to onset

preSamp=round(epochWin(1)*fs); % negative value
postSamp=round(epochWin(2)*fs);

erpTimes=(preSamp:postSamp)./fs;

erpData=[];

for ii=1:numel(eegCell)
    eeg=eegCell{ii}';
    onsets=silenceCell{ii};
    T = size(eeg,2);
    for kk=1:numel(onsets)
        idxStart=
    end

end
end