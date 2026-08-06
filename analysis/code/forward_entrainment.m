
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
% ERP CALCULATION (time-locked to silent onset)
% -------------------------------------------------------------------------
[erpData,erpTimes]=computeERPs(EEG.resp,silCell,cfg.fs,cfg.erpWindow);
% erpData: nChn x nTimes grand-average ERPs
%% ------------------------------------------------------------------------
% TRF CALCULATION (envelope + binary silence-onset feature)
% -------------------------------------------------------------------------
% build binary silence-onset regressor: same length as the envelope with a
% 1 at each detected silence onset sample.

onsetBinCell=cell(nStim,1);
for ii=1:nStim
    onsetVec_=zeros(numel(envCell{ii}),1);
    onsetVec_(silCell{ii})=1;
    onsetBinCell{ii}=onsetVec_;
end
clear onsetVec_
% NOTE: likely will end up clearing trf_config to avoid confusion, just
% need a dummy config with norm_envs, zscore_envs, zscore_eeg logial
% params to run rescale_trf_vars (which we probably should just change so
% it establishes default parameters when none provided).
[normEnvCell,normEEG]=rescale_trf_vars(envCell,EEG,trf_config);
normRespCell=normEEG.resp; %note: this seems needlessly complicated
[trfModel, lambdaOpt, statsObs, outerFoldLambdas]=computeTRF( ...
    normEnvCell, onsetBinCell, normRespCell, cfg.fs, ...
    cfg.trfLags, cfg.lambdaGrid);
fprintf('Optimal lambda per outer fold: %s\n',mat2str(outerFoldLambdas));
frpintf('Selected lambda for final fit: %g\n',lambdaOpt);
% extract useful vars from trf model
trfTimes=trfModel.t; % lag axis in ms
envTRF=squeeze(trfModel.w(1,:,:))'; % channels x lags
silenceTRF=squeeze(trfModel.w(2,:,:))'; % channels x lags


%% ------------------------------------------------------------------------
% ROI-AVERAGED PLOTS (averaged over cfg.roiElectrodes)
% -------------------------------------------------------------------------
% note, we already have them as indices, but could devise a function to map
% electrone names to indices if needed


roiIdx=cfg.roiElectrodes;

figure('Name', 'ROI-averaged responses to offsets');

ax1=subplot(2,1,1);
plot(1e3*erpTimes,mean(erpData(roiIdx,:),1),'LineWidth',1.5);
% xline(0,'')
xlabel('Time from silence onset (ms)'); ylabel('Amplitude'); % what units?
title('ROI ERP') %TODO: add topo marking selected electrodes

ax2=subplot(2,1,2);
plot(trfTimes, silenceTRF(roiIdx,:),'Linewidth', 1.5)
xlabel('Lag (ms)');ylabel('Amplitude (a.u)');
title('ROI TRF - silence onset feature')

linkaxes([ax1,ax2],'x')
xlim(ax2,[-100,500])

% NOTE: envelope TRF not time-locked to silences like ERP so maybe plot
% separately?
% subplot(3,1,3);
figure('Name', 'ROI-averaged envelope TRF')
plot(trfTimes, envTRF(roiIdx,:),'LineWidth', 1.5)
xlabel('Lag (ms)'); ylabel('Amplitude (a.u.)');
title('ROI TRF - envelope feature')
xlim([-100,500])



%TODO: add FFTs 
%% ------------------------------------------------------------------------
% BUTTERFLY + GFP + TOPO PLOTS
% -------------------------------------------------------------------------

plotButterflyGFPTopo(erpData, erpTimes, chanlocs, ...
    cfg.baselineWinERP, 'ERP (all channels)')

plotButterflyGFPTopo(envTRF, trfTimes, chanlocs, ...
    cfg.baselineWinTRF, 'TRF - envelope feature')

plotButterflyGFPTopo(silenceTRF, trfTimes, chanlocs, ...
    cfg.baselineWinTRF, 'TRF - silence onset feature')

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
    runStarts=find(d==1); runEnds=find(d==-1)-1;
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
nEpochs=sum(cellfun(@numel,silenceCell));

for ii=1:numel(eegCell)
    eeg=eegCell{ii}';
    onsets=silenceCell{ii};
    T = size(eeg,2);
    if ii==1
        %initialize epoch data array
        nChans =size(eeg,1);
        epochData=nan(nEpochs,nChans,numel(erpTimes));
        epochNum=1;
    end
    for kk=1:numel(onsets)
        % since preallocation in epochData ignores possibility of edge-case
        % epochs, still need to update the counter for those left-out
        % epochs
        epochNum=epochNum+1; 
        idxStart=onsets(kk)+preSamp;
        idxEnd=onsets(kk)+postSamp;
        % exclude epochs outside trial bounds -- note this automatically
        % excludes trial-start silent periods (unless no pre-onset times
        % specified in epochWin, but we need that for baseline-correction)
        if idxStart<1||idxEnd>T, continue; end
        epoch=eeg(:,idxStart:idxEnd);
        baseline=mean(epoch(:,erpTimes<=0),2);
        epochData(epochNum,:,:)=epoch-baseline;
    end
end
if ~exist("epochData",'var')||all(isnan(epochData(:)))
    warning('No valid epochs exracted!')
else
    erpData=squeeze(mean(epochData,1,'omitnan'));
end
end

function [trfModel, lambdaOpt, outerFoldStatsObs, outerFoldLambdas]=computeTRF( ...
envCell, onsetBinCell, eegCell, fs, lags, lambdaGrid)
% nested-cv lambda selection, then final dual-feature TRF fit
% does LOO cv for both inner and outer folds
%
% envCell{i}, onsetBinCell{i} : 1 x T_i vectors (envelope, binary onset)
% eegCell{i} : T_i x nChan EEG matrix
% fs: sampling rate (Hz)
% lags: [tmin tmax] in ms
% lambdaGrid: candidate ridge lambdas
%
% model: final TRF model struct (mTRFtrain output) fit on ALL trials at
% lambdaOpt
% lambdaOpt: lambda selected for the final fit
% outerFoldStatsObs: mTRFcrossval output statistics
% outerFoldLambdas: lambda chosen within each outer fold (diagnostic)

    nTrials=numel(eegCell);
    stim=cell(nTrials,1);
    for ii=1:nTrials
        stim{ii}=[envCell{ii}(:), onsetBinCell{ii}(:)];
    end
    outerPart=cvpartition(nTrials,'Leaveout');

    % optimize TRF lambda parameter using nested cv
    for oo=1:nTrials
        fprintf('outer fold %d of %d',oo,nTrials);
        trainIdx=training(outerPart,oo);
        trainStim=stim(trainIdx);
        trainResp=eegCell(trainIdx);

        innerStatsObs=mTRFcrossval(trainStim,trainResp,fs,1, ...
            lags(1),lags(2),lambdaGrid); %statsObs.r: [kfold x nlambda x nChan]
        % average over trials and electrodes
        meanR=squeeze(mean(mean(innerStatsObs.r,1),3)); % [ nlambda x 1]
        [~,bestLambdaIdx]=max(meanR); 
        % select lambda with highest overall R
        outerFoldStatsObs(oo)=innerStatsObs;
        outerFoldLambdas(oo)=lambdaGrid(bestLambdaIdx);
    end

    % train optimal lambda TRF model on entire dataset -- use median to
    % mitigate outliers
    lambdaOpt=median(outerFoldLambdas);
    %note: zero-padding biases TRF weights near tmin/tmax to be zero,
    %which might artificially 'erase' forward-entrainment signatures in the
    %TRF, but maybe consider the implications of data loss incurred by this
    trfModel=mTRFtrain(stim,eegCell,fs,1,lags(1),lags(2),lambdaOpt,'zeropad',0);

end

function winMask=computeGFPWindow(gfp, times, baselineWin)
% Flag time samples where GFP exceeds 2x the mean GFP measured in a
% baseline window.
%
% gfp: 1 x nTimes global field power
% times: 1 x nTimes axis (same units as baselineWin)
% baselineWin: [start stop] window of time used to compute reference level
%
% winMask: logical 1 x nTimes. true where gfp > 2 * mean(baseline gfp)

baselineIdx= times >= baselineWin(1) & times <= baselineWin(2);
baselineLevel=mean(gfp(baselineIdx));
winMask=gfp>2*baselineLevel;
end

function plotButterflyGFPTopo(data, times, chanlocs, baselineWin, titleStr)
% Butterfly plot + GFP trace + topoplot of the time-averaged weights within
% the GFP-define window (GFP > 2x baseline-window mean)
%
% data: nChn x nTimes matrix (ERP or single TRF feature)
% times: 1 x nTimes time axis (ms)
% chanlocs: EEGLAB chanlocs structure array
% baselineWin: [start stop] window defining the GFP reference level
% titleStr: figure title

gfp=std(data,0,1); % 1 x nTimes

winMask=computeGFPWindow(gfp, times, baselineWin);

figure('Name', [titleStr 'butterfly + GFP']);
% --- Butterfly + GFP ---

plot(times, data', 'Color', [.6 .6 .6],'LineWidth',.5); % plot all channels
hold on
plot(times,gfp,'k','Linewidth',2) % overlay GFP

if any(winMask)
    % determine how many separate supra-threshold windows exist
    dMask=diff([0 winMask 0]);
    winStarts=find(dMask==1);winEnds=find(dmask==-1)-1;
    nWindows=numel(winStarts);
    
    yl=ylim(ax1);
    for ww=1:nWindows
        patch(ax1, [times(winStarts(ww)) times(find(winMask, 'last')), ...
            times(winStarts(ww)) times(winEnds(ww))], ...
            [yl(1) yl(1) yl(2) yl(2)]);
    end
end
xlabel('Time (ms)'); ylabel('Amplitude/weights')
title([titleStr 'Butterfly + GFP'])

hold off


% --- Topoplot of time-averaged weights within GFP window ---
if any(winMask)
    for ww=1:nWindows
        figure('Name', sprintf('%s time-averaged topo %d of %d',titleStr,ww,nWindows))
        topoWeights=mean(data(:,winStarts(ww):winEnds(ww)));
        title(sprintf('Time-averaged topo weights %0.1fms - %0.1fms', ...
            times(winStarts(ww),times(winEnds(ww)))))
        topoplot(topoWeights,chanlocs,'electrodes','on')
        colorbar;
    end
else
    disp('No supra-threshold GFP windows to time average for topoplot.')
end
end