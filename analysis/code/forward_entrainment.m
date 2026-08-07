
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


cfg.doGroupAverage=true;

cfg.overwrite=0; 

%% ------------------------------------------------------------------------
% LOAD DATA + BATCH LOOP
% -------------------------------------------------------------------------
subjs=[24:25];
allResults=cell(numel(subjs),1);

for subjIdx=1:numel(subjs)
    subj=subjs(subjIdx);
    trf_analysis_params % loc_file, trf_config, prepocess_config     
    % NOTE: we might want to change this (for no other reason than it's
    % kinda clunky even though it works)
    if subjIdx==1&&~exist('chanlocs','var')
        load(loc_file)
    end
    resultsCfg=buildResultsCacheKey(subj,cfg,preprocess_config,trf_config);
    try
        results=load_checkpoint(resultsCfg);
        fprintf('loaded pre-existing results for subj %d\n',subj)
    catch
        fprintf(['saved results file for' ...
            ' subj %d not found, computing from start.\n'], subj)
        results=runSubjectAnalysis(subj,cfg,preprocess_config, trf_config);
        % save_checkpoint(results,resultsCfg,cfg.overwrite)
    end
    allResults{subjIdx}=results;
    clear results
end

%% ------------------------------------------------------------------------
% ROI-AVERAGED (over cfg.roiElectrodes) TIMECOURSES, 
%               BUTTERFLY, GFP, TOPO PLOTS 
% 
% TODO: FFTs(?)
% -------------------------------------------------------------------------
% plotButterflyGFPTopo assumes times in ms
%TODO: GET CONDVEC???????
condVec=allResults{1}.condVec;
for subjIdx=1:numel(subjs)
    subj=subjs(subjIdx);
    plot_subject_results(allResults(subjIdx),chanlocs, condVec, cfg,['subj %d' subj])
end
%% ------------------------------------------------------------------------
% COMPUTE GRAND AVERAGE RESULTS + plot
% -------------------------------------------------------------------------
if cfg.doGroupAverage
    grand=computeGrandAverage(allResults,chanlocs);
    plot_subject_results(grand,chanlocs,condVec,cf)
end


%% ------------------------------------------------------------------------
% Local Functions
% -------------------------------------------------------------------------
function results=runSubjectAnalysis(subj,cfg,preprocess_config,trf_config)
% subj: scalar, subject ID number
% cfg: struct, with analysis params
% wrapper to do silence detection, ERP analysis, and TRF analysis on a
% single-subject's data

    pp_checkpoint_=load_checkpoint(preprocess_config);
    EEG=pp_checkpoint_.preprocessed_eeg;
    envCell=load_stim_cell(trf_config.paths.envelopesFile, EEG.cond, EEG.trials);

    condVec=EEG.cond; %nStim x 1 vector of condition index


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
    [erpData,erpTimes]=computeERPs(EEG.resp,silCell,condVec,cfg.fs,cfg.erpWindow);
    % erpData: nConditions x nChn x nTimes grand-average ERPs
%% ------------------------------------------------------------------------
% TRF CALCULATION (envelope + binary silence-onset feature)
% -------------------------------------------------------------------------
% build binary silence-onset regressor: same length as the envelope with a
% 1 at each detected silence onset sample.
    %TODO: add try-catch in case trf fitting fails for some reason
    onsetBinCell=cell(nStim,1);
    for ii=1:nStim
        onsetVec_=zeros(numel(envCell{ii}),1);
        onsetVec_(silCell{ii})=1;
        onsetBinCell{ii}=onsetVec_;
    end

    % NOTE: likely will end up clearing trf_config to avoid confusion, just
    % need a dummy config with norm_envs, zscore_envs, zscore_eeg logial
    % params to run rescale_trf_vars (which we probably should just change so
    % it establishes default parameters when none provided).
    [normEnvCell,normEEG]=rescale_trf_vars(envCell,EEG,trf_config);
    normRespCell=normEEG.resp; %note: this seems needlessly complicated

    % TODO: only do this step when conditions are not separate? and define
    % lambda from condition-agnostic fit file? OR just leave it alone and
    % do the crossvalidation per condition to see if lambda changes or not
    [trfModels, lambdaOpt, statsObs, outerFoldLambdas]=computeTRF( ...
        normEnvCell, onsetBinCell, normRespCell, condVec, cfg.fs, ...
        cfg.trfLags, cfg.lambdaGrid);
    fprintf('Optimal lambda per outer fold: %s\n',mat2str(outerFoldLambdas));
    fprintf('Selected lambda for final fit: %g\n',lambdaOpt);
    % extract useful vars from trf model -- NOTE: I don't like this because
    % information already in trfModel but going with it for now, change if
    % it causes memory issues
    % trfTimes=trfModels(1).t; % lag axis in ms
    % envTRF=squeeze(trfModels.w(1,:,:))'; % channels x lags
    % silenceTRF=squeeze(trfModels.w(2,:,:))'; % channels x lags
    results=struct("subj",subj, ...
    "silCell", silCell, ...
    "erpData", erpData, ...
    "erpTimes", erpTimes, ...
    "onsetBinCell",onsetBinCell, ...
    "trfModels", trfModels, ...
    "lambdaOpt", lambdaOpt,...
    "statsObs",statsObs, ...
    "condVec",condVec);
    % "trfTimes", trfTimes, ...
    % "envTRF", envTRF, ...
    % "silenceTRF", silenceTRF, ...
    
end

function grand = computeGrandAverage(allResults, chanlocs)
% TODO: RE-FACTOR TO SEPARATE RESULTS BY CONDITION
    nSubj=numel(allResults);

    % sanity checks: all results should be able to share time axis
    erpTimesRef=allResults{1}.erpTimes;
    trfTimesRef=allResults{1}.trfTimes;
    nChanRef=numel(chanlocs);

    for ss=2:nSubj
        assert(isequal(allResults{ss}.erpTimes,erpTimesRef), ...
            'Subj %d ERP time axis does not match Subj 1')
        assert(isequal(allResults{ss}.trfTimes,trfTimesRef), ...
            'Subj %d trfTimes does not match subj 1')
        assert(size(allResults{ss}.erpData,1)==nChanRef & ...
            size(allResults{ss}.envTRF,1)==nChanRef & ...
            size(allResults{ss}.silenceTRF,1)==nChanRef)
    end

    % preallocate stack matrices
    erpStack=nan(nChanRef,numel(erpTimesRef),nSubj);
    envTRFStack=nan(nChanRef,numel(trfTimesRef),nSubj);
    silenceTRFStack=nan(nChanRef,numel(trfTimesRef),nSubj);
    for ss=1:nSubj
        erpStack(:,:,ss)=allResults.erpData;
        envTRFStack(:,:,ss)=allResults.envTRF;
        silenceTRFStack(:,:,ss)=allResults.silenceTRF;
    end

    grand.erpData=mean(erpStack,3);
    grand.envTRF=mean(envTRFStack,3);
    grand.silenceTRF=mean(silenceTRFStack,3);

end


function plot_subject_results(results, chanlocs, condVec, cfg, labelStr)
    roiIdx=cfg.roiElectrodes;

    for condIdx=1:numel(condVec)
        %TODO: UNPACK BELOW CORRECTLY
        trfTimes=results.trfModels(1).t; % lag axis in ms
        % envTRF=squeeze(trfModels.w(1,:,:))'; % channels x lags
        % silenceTRF=squeeze(trfModels.w(2,:,:))'; % channels x lags
    
    
        figure('Name', ['ROI-averaged responses to offsets - ' labelStr]);
        ax1=subplot(2,1,1);
        plot(1e3*results.erpTimes,mean(results.erpData(condIdx,roiIdx,:),1),'LineWidth',1.5);
        xline(0,'k--')
        xlabel('Time from silence onset (ms)'); ylabel('Amplitude'); % what units?
        title(['ROI ERP - ' labelStr]) %TODO: add topo marking selected electrodes
        
        ax2=subplot(2,1,2);
        % silent trf is second feature
        plot(trfTimes, mean(results.trfModels(condIdx).w(2,roiIdx,:)'),'Linewidth', 1.5)
        xline(0, 'k--')
        xlabel('Lag (ms)');ylabel('Amplitude (a.u)');
        title(['ROI TRF - silence onset feature' labelStr])
        
        linkaxes([ax1,ax2],'x')
        xlim(ax2,[-100,500])
        
        % NOTE: envelope TRF not time-locked to silences like ERP so maybe plot
        % separately?
        % subplot(3,1,3);
        figure('Name', 'ROI-averaged envelope TRF')
        plot(results.trfTimes, results.trfModels(condIdx).w(roiIdx,:)','LineWidth', 1.5)
        xlabel('Lag (ms)'); ylabel('Amplitude (a.u.)');
        title('ROI TRF - envelope feature')
        xlim([-100,500])
        
        
        
        %TODO: add FFTs 
        
        plotButterflyGFPTopo(squeeze(results.erpData(condIdx,:,:)), 1e3*results.erpTimes, chanlocs, ...
            cfg.baselineWinERP, sprintf('ERP (all channels) - %s - cond: %d',labelStr,condIdx))
        
        plotButterflyGFPTopo(squeeze(results.trfModels(condIdx).w(1,:,:)), ...
            results.trfTimes, chanlocs, ...
            cfg.baselineWinTRF, sprintf('TRF - envelope feature - %s - cond: %d', labelStr,condIdx));

        
        plotButterflyGFPTopo(squeeze(results.trfModels(condIdx).w(2,:,:)), ...
            results.trfTimes, chanlocs, ...
            cfg.baselineWinTRF, sprintf('TRF - silence onset feature - %s - cond: %d', labelStr,condIdx))
    end
end

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

function [erpData,erpTimes]=computeERPs(eegCell,silenceCell,condVec,fs,epochWin)
% Epoch EEG around silence onsets, pool across stimuli, average across
% epochs
%
% eegCell{i} : T_i x nChns EEG matrix for stimulus i
% silenceCell{i} : sample indices of silence onsets for stimulus i
% condVec: nStim x 1 vector of condition-label index
% fs: sampling rate (Hz)
% epochWin : [preSec postSec] window around onsets
%
% erpData : nCond x nChns x nTimes (baseline-corrected, 
% condition-specific, averaged across epochs)
%
% erpTimes : 1 x nTimes time axis in seconds relative to onset


% same for all conditions
preSamp=round(epochWin(1)*fs); % negative value
postSamp=round(epochWin(2)*fs);

erpTimes=(preSamp:postSamp)./fs;
nEpochs=sum(cellfun(@numel,silenceCell));

nCond=numel(unique(condVec));

for ii=1:numel(eegCell)
    eeg=eegCell{ii}';
    condIdx=condVec(ii);
    onsets=silenceCell{ii};
    T = size(eeg,2);
    if ii==1
        %initialize epoch data array
        nChans =size(eeg,1);
        epochData=nan(nCond,nEpochs,nChans,numel(erpTimes));
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
        %baseline correction on erps
        baseline=mean(epoch(:,erpTimes<=0),2);
        epochData(condIdx,epochNum,:,:)=epoch-baseline;
    end
end
if ~exist("epochData",'var')||all(isnan(epochData(:)))
    warning('No valid epochs exracted!')
else
    erpData=squeeze(mean(epochData,2,'omitnan'));
end
end

function [trfModels, lambdaOpt, outerFoldStatsObs, outerFoldLambdas]=computeTRF( ...
envCell, onsetBinCell, eegCell, condVec, fs, lags, lambdaGrid)
% nested-cv lambda selection, then final dual-feature TRF fit
% does LOO cv for both inner and outer folds
%
% envCell{i}, onsetBinCell{i} : 1 x T_i vectors (envelope, binary onset)
% eegCell{i} : T_i x nChan EEG matrix
% condVec: nTrias x 1 vector 
% fs: sampling rate (Hz)
% lags: [tmin tmax] in ms
% lambdaGrid: candidate ridge lambdas
%
% trfModel: (nCond x 1) final TRF model struct (mTRFtrain output) fit on ALL trials at
% 
% lambdaOptCell: (cond x 1) lambda selected for the final fit
% outerFoldStatsObs: (cond x nTrials) struct mTRFcrossval output statistics
% outerFoldLambdas: lambda chosen within each outer fold (diagnostic)
% note it is assumed all conditions have the same number of trials

    nCond=numel(unique(condVec));
    for condIdx=1:nCond %TODO: decide how to package mote, lambda opt, outerFoldStatsObs, and outerFoldLambdas for output!
        condMask=condVec==condIdx;
        condEegCell=eegCell(condMask);
        condEnvCell=envCell(condMask);
        condOnsetBinCell=onsetBinCell(condMask);
        nTrials=numel(condEegCell); %TODO: validate that this 
        % is same across conditions (assert)
        condStim=cell(nTrials,1);
        for ii=1:nTrials
            condStim{ii}=[condEnvCell{ii}(:), condOnsetBinCell{ii}(:)];
        end
        outerPart=cvpartition(nTrials,'Leaveout');
    
        % optimize TRF lambda parameter using nested cv
        for oo=1:nTrials
            fprintf('outer fold %d of %d (condition %d of %d)', ...
                oo,nTrials,condIdx,nCond);
            trainIdx=training(outerPart,oo);
            trainStim=condStim(trainIdx);
            trainResp=condEegCell(trainIdx);
    
            innerStatsObs=mTRFcrossval(trainStim,trainResp,fs,1, ...
                lags(1),lags(2),lambdaGrid); %statsObs.r: [kfold x nlambda x nChan]
            if condIdx==1 && oo==1
            % pre-allocate
                statsObsFieldnames=fieldnames(innerStatsObs);
                outerFoldStatsObs=cell2struct( ...
                    cell(numel(statsObsFieldnames),nCond,nTrials), ...
                    statsObsFieldnames);
                outerFoldLambdas=nan(nCond,nTrials);
            end
            % average over trials and electrodes
            meanR=squeeze(mean(mean(innerStatsObs.r,1),3)); % [ nlambda x 1]
            [~,bestLambdaIdx]=max(meanR);
            % select lambda with highest overall R
            outerFoldStatsObs(condIdx,oo)=innerStatsObs;
            outerFoldLambdas(condIdx,oo)=lambdaGrid(bestLambdaIdx);
        end
    
        % train optimal lambda TRF model on entire dataset -- use median to
        % mitigate outliers
        lambdaOpt=median(outerFoldLambdas(condIdx,:));
        %note: zero-padding biases TRF weights near tmin/tmax to be zero,
        %which might artificially 'erase' forward-entrainment signatures in the
        %TRF, but maybe consider the implications of data loss incurred by this
        condTrfModel=mTRFtrain(condStim,condEegCell,fs,1,lags(1),lags(2),lambdaOpt,'zeropad',0);
        if condIdx==1
            % "pre"-allocate
            condTrfModelFieldnames=fieldnames(condTrfModel);
            trfModels=cell2struct(cell(numel(condTrfModelFieldnames),nCond,1),condTrfModelFieldnames);
        end
        trfModels(condIdx)=condTrfModel;
    end

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
    winStarts=find(dMask==1);winEnds=find(dMask==-1)-1;
    nWindows=numel(winStarts);
    
    yl=ylim(gca());
    for ww=1:nWindows
        patch([times(winStarts(ww)) times(winEnds(ww)), ...
            times(winEnds(ww)) times(winStarts(ww))], ...
            [yl(1) yl(1) yl(2) yl(2)], [1 .9 .6], ...
            'FaceAlpha', 0.3, 'EdgeColor','none');
    end
end
xlabel('Time (ms)'); ylabel('Amplitude/weights')
title([titleStr 'Butterfly + GFP'])

hold off


% --- Topoplot of time-averaged weights within GFP window ---
if any(winMask)
    for ww=1:nWindows
        figure('Name', sprintf('%s time-averaged topo %d of %d',titleStr,ww,nWindows))
        topoWeights=mean(data(:,winStarts(ww):winEnds(ww)),2);
        title(sprintf('Time-averaged topo weights %0.1fms - %0.1fms', ...
            times(winStarts(ww)),times(winEnds(ww))))
        topoplot(topoWeights,chanlocs,'electrodes','on')
        colorbar;
    end
else
    disp('No supra-threshold GFP windows to time average for topoplot.')
end
end

function resultsCfg=buildResultsCacheKey(subj,cfg,preprocess_config,trf_config)
%
    resultsCfg=struct( ...
        'subj',subj, ...
        'fs', cfg.fs, ...
        'silenceThresh', cfg.silenceThresh, ...
        'minSilenceDur', cfg.minSilenceDur, ...
        'erpWindow', cfg.erpWindow, ...
        'trfLags', cfg.trfLags, ...
        'lambdaGrid', cfg.lambdaGrid, ...
        'preprocess_config', preprocess_config, ...
        'trf_config', trf_config); %trf config might be confusing in 
    % future, consider only keeping the normalization parameters since
    % that's all we end up using
end