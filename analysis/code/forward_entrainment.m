
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

dbstop if error
cfg=[];
cfg.configType='fEntrainment';
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

cfg.overwrite=0; %TODO: CHECK IF THIS WORKS!

cfg.doExportFigures=true;


%% ------------------------------------------------------------------------
% LOAD DATA + BATCH LOOP
% -------------------------------------------------------------------------
subjs=[24:38];
allResultsCell=cell(numel(subjs),1);

for subjIdx=1:numel(subjs)
    subj=subjs(subjIdx);
    trf_analysis_params % loc_file, trf_config, prepocess_config
    clear do_nulltest show_tuning_curves train_params
    % NOTE: we might want to explicitly define pre-requisite configs in-script
    % (for no other reason than current process 
    % kinda clunky even though it works)
    if subjIdx==1&&~exist('chanlocs','var')
        load(loc_file)
    end
    %%
    subjResultsCfg=buildResultsCacheKey(subj,cfg,preprocess_config,trf_config);
    % above function just adds paths and ensures nested configs are forced
    % to be hashable -- still needs to be made hashable itself. As a
    % result, consider renaming function as it seems to imply that output
    % is ready-to-hash
    % actually, progressLoad runs buildHashableConfig internally so dont
    % even really need to hash it..
    
    checkpointData=progressLoad(subjResultsCfg);
    noPreloadData=isempty(checkpointData);
    if noPreloadData
        fprintf(['saved results file for' ...
            ' subj %d not found, computing from start.\n'], subj)
        results=runSubjectAnalysis(subj,cfg,preprocess_config, trf_config);
        progressSave(results,subjResultsCfg,cfg.overwrite)
        checkpointData.results=results;
    else
        fprintf('loaded pre-existing results for subj %d\n',subj)
    end
    allResultsCell{subjIdx}=checkpointData.results;
    clear checkpointData subjResultsCfg
end

%% ------------------------------------------------------------------------
% ROI-AVERAGED (over cfg.roiElectrodes) TIMECOURSES, 
%               BUTTERFLY, GFP, TOPO PLOTS 
% PLOT -> EXPORT -> CLOSE (unless cfg.doExportFigures=false) 
%  SAFETY: figures stay open if cfg.doExportFigures=false, avoid if
%  plotting more than one subject
% TODO: FFTs(?)
% -------------------------------------------------------------------------
% plotButterflyGFPTopo assumes times in ms
for subjIdx=1:numel(subjs)
    subj=subjs(subjIdx);
    plot_subject_results(allResultsCell{subjIdx},chanlocs, ...
        cfg,sprintf('Subj%0d', subj))
    % EXPORT SUBJ FIGS
    if cfg.doExportFigures
        subjFigDir=fullfile(boxdir_mine,'figures',cfg.configType,sprintf('Subj%02d',subj));
        subjResultsCfg=buildResultsCacheKey(subj,cfg,preprocess_config,trf_config);
        exportFiguresToFolder(subjFigDir,subjResultsCfg);
        clear subjResultsCfg
    end
    fprintf('Plots for subj %02d done. Closed: %d\n',subj, cfg.doExportFigures)
    
end


%% ------------------------------------------------------------------------
% COMPUTE GRAND AVERAGE RESULTS + plot
% -------------------------------------------------------------------------
if cfg.doGroupAverage
    grand=computeGrandAverage(allResultsCell,chanlocs);
    plot_subject_results(grand,chanlocs,cfg,'Grand-Average')
    if cfg.doExportFigures
        grandAvgFigDir=fullfile(boxdir_mine,'figures',cfg.configType,'Grand Average');
        % note that preprocess_config & trf_config are from a single
        % subject but buildResultsCacheKey removes subject label... subject
        % specific idiosyncracies in config might be problematic here
        % though....
        grandAvgResultsCfg=buildResultsCacheKey(subjs,cfg,preprocess_config,trf_config);
        exportFiguresToFolder(grandAvgFigDir,grandAvgResultsCfg)
        clear subjResultsCfg
    end

end


%% ------------------------------------------------------------------------
% Local Functions
% -------------------------------------------------------------------------
function results=runSubjectAnalysis(subj,cfg,preprocess_config,trf_config)
% subj: scalar, subject ID number
% cfg: struct, with analysis params
% wrapper to do silence detection, ERP analysis, and TRF analysis on a
% single-subject's data

    pp_checkpoint_=progressLoad(preprocess_config);
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
    % extract clean matrices from bloated trfModels
    nCond=numel(unique(condVec));
    nChans=size(EEG.resp{1},2);
    nLags=numel(trfModels(1).t);

    envTRF=nan(nCond,nLags,nChans);
    silTRF=nan(nCond,nLags,nChans);
    trfTimes=trfModels(1).t; % lag axis in ms

    for condIdx=1:nCond
        envTRF(condIdx,:,:)=squeeze(trfModels(condIdx).w(1,:,:));
        silTRF(condIdx,:,:)=squeeze(trfModels(condIdx).w(2,:,:));
    end
    % NOTE: intentionally avoid saving trfModels
    results=struct("subj",subj, ...
    "silCell", {silCell}, ...
    "erpData", erpData, ...
    "erpTimes", erpTimes, ...
    "onsetBinCell",{onsetBinCell}, ...
    "envTRF", envTRF, ...
    "silTRF", silTRF, ...
    "trfTimes",trfTimes, ...
    "lambdaOpt", lambdaOpt,...
    "statsObs",statsObs, ...
    "condVec",condVec);
    % "trfTimes", trfTimes, ...
    % "envTRF", envTRF, ...
    % "silenceTRF", silenceTRF, ...
    
end

function grand = computeGrandAverage(allResults, chanlocs)
% GRAND MUST INHERET ALL RESULTS TO MATCH STRUCTURE OF PLOTTING FUNCTION
% LATER
    nSubj=numel(allResults);

    % sanity checks: all results should be able to share time axis
    erpTimes=allResults{1}.erpTimes;
    trfTimes=allResults{1}.trfTimes;
    nChan=numel(chanlocs);
    nCond=numel(unique(allResults{1}.condVec));

    for ss=2:nSubj
        assert(isequal(allResults{ss}.erpTimes,erpTimes), ...
            'Subj %d ERP time axis does not match Subj 1',ss)
        
        assert(isequal(allResults{ss}.trfTimes,trfTimes), ...
            'Subj %d trfTimes does not match subj 1',ss)
        
        assert(size(allResults{ss}.erpData,2)==nChan& ...
            size(allResults{ss}.envTRF,3)==nChan& ...
            size(allResults{ss}.silTRF,3)==nChan, ...
            'number of expected channels not consistent')
    end

    % preallocate stack matrices
    erpStack=nan(nCond,nChan,numel(erpTimes),nSubj);
    envTRFStack=nan(nCond,numel(trfTimes),nChan,nSubj);
    silenceTRFStack=nan(nCond,numel(trfTimes),nChan,nSubj);
    for ss=1:nSubj
        erpStack(:,:,:,ss)=allResults{ss}.erpData;
        envTRFStack(:,:,:,ss)=allResults{ss}.envTRF; %envTRF
        silenceTRFStack(:,:,:,ss)=allResults{ss}.silTRF; %silenceTRF
    end

    % create grand results struct
    grand.sbuj='Grand-Average';
    grand.erpTimes=erpTimes;
    grand.trfTimes=trfTimes;
    % average across subjects
    grand.erpData=mean(erpStack,4);
    grand.envTRF=mean(envTRFStack,4);
    grand.silTRF=mean(silenceTRFStack,4);

end


function plot_subject_results(results, chanlocs, cfg, subjLabelStr)
% wrapper function for ROI-averaged ERP/TRF results + plotButterflyGFPTopo
% results: 1x1 struct containing results
    roiIdx=cfg.roiElectrodes;
    nCond=size(results.erpData,1);
    trfTimes=results.trfTimes; % lag axis in ms

    for condIdx=1:nCond
        % condMask=condVec==condIdx; % nStim x1 logical nvm dont need
        
        condRoiSilERP=squeeze(mean(mean(results.erpData(condIdx,roiIdx,:),1),2)); % lags x 1
        condRoiSilTRF=squeeze(mean(results.silTRF(condIdx,:,roiIdx),3))'; % lags x 1
        condRoiEnvTRF=squeeze(mean(results.envTRF(condIdx,:,roiIdx),3))'; %lags x 1
        
        % define a clean prefix for subject/condition
        figPrefix=sprintf('%s | Cond%d | ',subjLabelStr,condIdx);
        figName=[figPrefix 'ROI-averaged timecourses'];
        figure('Name', figName);
        ax1=subplot(3,1,1);
        plot(1e3*results.erpTimes,condRoiSilERP,'LineWidth',1.5);
        xline(0,'k--')
        xlabel('Time from silence onset (ms)'); ylabel('Amplitude'); % what units?
        title('silERP') %TODO: add topo marking selected electrodes
        
        ax2=subplot(3,1,2);
        plot(trfTimes, condRoiSilTRF,'Linewidth', 1.5)
        xline(0, 'k--')
        xlabel('Lag (ms)');ylabel('Amplitude (a.u)');
        title('silTRF ')
        
        
        % NOTE: envelope TRF not time-locked to silences like ERP so maybe plot
        % separately?
        ax3= subplot(3,1,3);
        plot(trfTimes, condRoiEnvTRF,'LineWidth', 1.5)
        xlabel('Lag (ms)'); ylabel('Amplitude (a.u.)');
        title('envTRF')
        xlim([-100,500])

        sgtitle(figName)
        linkaxes([ax1,ax2,ax3],'x')
        xlim(ax3,[-100,500])
        
        
        
        %TODO: add FFTs 
        
        plotButterflyGFPTopo(squeeze(results.erpData(condIdx,:,:)), 1e3*results.erpTimes, chanlocs, ...
            cfg.baselineWinERP, [figPrefix 'silERPs'])
        
        plotButterflyGFPTopo(squeeze(results.envTRF(condIdx,:,:))', ...
            trfTimes, chanlocs, ...
            cfg.baselineWinTRF, [figPrefix 'envTRFs']);

        
        plotButterflyGFPTopo(squeeze(results.silTRF(condIdx,:,:))', ...
            trfTimes, chanlocs, ...
            cfg.baselineWinTRF, [figPrefix 'silTRF']);
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
%       trfModel weights have (feat x time x chans), 
%           feats order: (1) envelope, (2) onset  
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
% Butterfly plot + GFP trace + topoplots of the time-averaged weights within
% the GFP-define windows (GFP > 2x baseline-window mean)
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
    % create ONE figure to put all topoplots into
    figTopo=figure('Name', [titleStr ' | Topoplots'], 'NumberTitle', 'off');

    % use 'flow' tiled-layout to automatically arrrange subplots in a grid
    tlo=tiledlayout(figTopo, 'flow','TileSpacing','compact','Padding','compact');
    title(tlo, [titleStr ' | GFP Supra-threshold Windows'])
    for ww=1:nWindows
        nexttile(tlo); % add new tile automatically
        topoWeights=mean(data(:,winStarts(ww):winEnds(ww)),2);

        topoplot(topoWeights,chanlocs,'electrodes','on');
        subtitle(sprintf(' %0.1fms - %0.1fms', ...
            times(winStarts(ww)),times(winEnds(ww))))
        colorbar;
    end
else
    disp('No supra-threshold GFP windows to time average for topoplot.')
    %TODO: IT WOULD BE MORE NICER IF THIS RETURNED A FIGURE WITH THAT
    %MESSAAGE ON IT BUT OTHERWISE EMPTY
end
end

function resultsCfg=buildResultsCacheKey(subj,cfg,preprocess_config,trf_config)
%   not yet hashable, but ensures nested configs are hashable

    hashablePpConfig=buildHashableConfig(preprocess_config);
    hashableTrfConfig=buildHashableConfig(trf_config);

    global boxdir_mine
    if isscalar(subj)
        output_dir=sprintf('%s/analysis/forward_entrainment/%02d',boxdir_mine,subj);
    else
        output_dir=sprintf('%s/analysis/forward_entrainment/Grand Average',boxdir_mine);
    end
    paths=struct('output_dir',output_dir);
    resultsCfg=struct( ...
        'subj',subj, ...,
        'configType',cfg.configType, ...
        'fs', cfg.fs, ...
        'silenceThresh', cfg.silenceThresh, ...
        'minSilenceDur', cfg.minSilenceDur, ...
        'erpWindow', cfg.erpWindow, ...
        'trfLags', cfg.trfLags, ...
        'lambdaGrid', cfg.lambdaGrid, ...
        'preprocess_config', hashablePpConfig, ...
        'trf_config', hashableTrfConfig, ...
        'paths', paths); %trf config might be confusing in 
    % future, consider only keeping the normalization parameters since
    % that's all we end up using
end

function exportFiguresToFolder(targetFolder, varargin)
% EXPORTFIGURESTOFOLDER Exports all open figures to a specific folder,
% optionally saves a metadata struct, and CLOSES all figures to free RAM
%
% Usage:
%   export FiguresToFolder(targetFolder)
%   exporrtFiguresToFolder(targetFolder,metadataStruct)

% ensure target directory exists
if ~exist(targetFolder, 'dir')
    mkdir(targetFolder);
end

% find all open figures
allFigs=findobj('Type','figure');
nFigs=numel(allFigs);

if nFigs==0
    fprintf('No open figures to export to %s.\n',targetFolder)
    return;
end


fprintf('Exporting %d figures to %s...\n',nFigs,targetFolder)

% loop thru figs and export

for ii=1:nFigs
    fig=allFigs(ii);
    figName=fig.Name;
    % clean the figure name to make it a valid, readablefilename
    cleanName=strrep(figName,' | ','_');
    cleanName=strrep(cleanName,' ','_');
    % note: not used in this script but probably good to generally avoid OS
    % issues with unacceptable filenames
    cleanName=strrep(cleanName,':','');
    cleanName=strrep(cleanName,'/','_');
    
    % export as high-res PNG
    exportgraphics(fig, fullfile(targetFolder, [cleanName '.png']), ...
        'Resolution', 300);


end
% save metadata if provided
if ~isempty(varargin)
    metadataStruct=varargin{1};
    metaFile=fullfile(targetFolder, 'metadata.mat');
    save(metaFile,'metadataStruct');
    fprintf('Saved metadata to %s\n', metaFile);
end
% close all figures to free up RAM
close(allFigs);
fprintf('Closed %d figures to free memory.\n',nFigs);
end