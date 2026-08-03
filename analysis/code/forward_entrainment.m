%% FORWARD ENTRAINMENT ANALYSIS
% ------------------------------------------
% Searches for signatures of forward entrainment (persistent neural
% activity during silences) using two complementary analyses:
% 
% 1. Channel-wise ERPs time-locked to the onset of detected silences
% 2. Dual-feature TRFs (envelope + binary silece-onset regressor) fir with
% ridge parameter (lambda) chosen via nested cross-validation, then refit
% on the full dataset at the optimal lambda
% ------------------------------------------------

clear; clc; close all;
%% -------------------------------------------------------------------
% USER CONFIG
% -----------------------------------------------------------------------


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
trf_analysis_params
pp_checkpoint_=load_checkpoint(preprocess_config);
trf_checkpoint_=load_checkpoint(trf_config);

% define chanlocs by running trf_analysis
load(loc_file)
