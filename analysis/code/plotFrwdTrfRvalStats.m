% plotFwdTrfRvalStats
% PLOTS:
% 1. per-subject pretty scat (overlaid boxplot + electrode-connecting lines 
% across conditons) of forward-model encoding accuracy accuracy
% 2. across-subjs pretty scat (+ boxplot OR violin [???] +
% subject-connecting lines), averaging r across anointed electrodes
%% -- LOAD rVALS

cleanWorkspace
checkInputSubjs

metadata=struct('trf_config',[],'preprocess_config',[]);
baseCfg.configType='frwdTrfRvalStats';

baseCfg.nTrialsPerCond=25; %TODO: what if we wanna plot all conditions?
baseCfg.nChans=128;
baseCfg.nCond=3;
allSubjRvals=nan(ss,baseCfg.nCond,baseCfg.nTrialsPerCond,baseCfg.nChans);
doFigExport=true;
for ss=1:numel(subjs)
    subj=subjs(ss);
    setTrfAnalysisParams
    clear show_tuning_curves
    metadata(ss).trf_config=trf_config; metadata(ss).preprocess_config=preprocess_config;
    allSubjRvals(ss,:,:,:)=statsLoadWrapper(trf_config);
    clear subj trf_config preprocess_config
end

%% -- PLOT SUBJECT-LEVEL PRETTY SCATS (BOXPLOT)

for ss=1:numel(subjs)
    subj=subjs(ss);
    subjCfg_=baseCfg;
    subjCfg_.experiment=experimentsBySubjNum{ss}; % note: not assuming all subjects from same experiment
    subjCfg_=addResultsPaths(subjCfg_,subj);
    condLabels_=metadata(ss).trf_config.conditions;
    % average across trials before plotting
    fig=prettyScat(squeeze(mean(allSubjRvals(ss,:,:,:),3)),condLabels_,1,'scatBox');
    title(sprintf('Subj %d TRF Accuracies',subj))
    if doFigExport
        exportFiguresToFolder(subjCfg_.paths.figDir,metadata(ss),fig,0)
    end
    clear subj condLabels_ subjCfg_
end


%% -- PLOT ACROSS SUBJECTS PRETTY SCATS (VIOLIN)

% HERE WE SHOULD ASSUME ALL SUBJECTS BELONG TO SAME EXPERIMENT
assert(isscalar(unique(experimentsBySubjNum)),'nonHomogenousExperiment: all subjects must be from same experiment!')
baseCfg.experiment=experimentsBySubjNum{1};
condLabels=metadata(1).trf_config.conditions;
baseCfg.selectChns=[54 55 56 61 62 63 106 107 108 115 116 117]; %'all' works
baseCfg.acrossSubjsChnsPlotted=normalizeChnsSelection(baseCfg.selectChns);
baseCfg=addResultsPaths(baseCfg,subjs,'allSubjs');

acrossSubjsRvals=mean(allSubjRvals(:,:,:,baseCfg.selectChns),[3,4]);

fig=prettyScat(acrossSubjsRvals,condLabels,2,'scatViolin');
title('Prediction Accuracy Across Subjects (auditory-responsive channels)')
if doFigExport
    exportFiguresToFolder(baseCfg.paths.figDir,metadata,fig,0)
end



%% HELPERS
function subjR=statsLoadWrapper(trf_config)
% statsLoadWrapper(trf_config)
% NOTE: it would be more convenient if something similar to this were the
% default behavior / ouptut of progressLoad to avoid having to do this
% unpacking of variables in every script (def a Claude-solvable issue)
matData=progressLoad(trf_config);
nCond=numel(matData.stats_obs);
[nTrialsPerCondition,~,nChans]=size(matData.stats_obs(1).r);
subjR=nan(nCond,nTrialsPerCondition,nChans);
for cc=1:nCond
subjR(cc,:,:)=squeeze(matData.stats_obs(cc).r);
end

end
function fig=prettyScat(D,condLabels,condDim,plotType)
% PRETTYSCAT scatterplot with per-sample connecting lines and boxplots OR violin plots.
%
% pretty_scat(D, cond_dim)
%
% D - data matrix of any dimensionality
% cond_dim - index of the dimension to use at the x-axis (condition/pairs)
%
% Permutes D so that cond_dim is last, then reshapes to [n_smaples x
% n_cond] and plots grey connecting lines, scatter points, and boxplots.

if nargin<4
    plotType='scatBox'; % 'scatBox' or 'scatViolin'
end

sz = size(D);
n_cond = sz(condDim);

% Permute so that condition dimension is last
if condDim ~= numel(sz)
    dim_ord = 1:numel(sz);
    dim_ord = [dim_ord(dim_ord~=condDim), condDim];
    D = permute(D, dim_ord);
    sz = size(D);
end

n_per_cc = prod(sz(1:end-1));

fig=figure('Units','inches','Position',[0 0 8 5]); hold on

% Make everything grey
point_color = [0.5 0.5 0.5];
line_color = [0.7 0.7 0.7];

% Reshape so we can plot across conditions easily
Dmat = reshape(D, [n_per_cc, n_cond]);

% --- 1) Draw connecting lines across each row (each sample)
for ii = 1:n_per_cc
    plot(1:n_cond, Dmat(ii,:), '-', 'Color', line_color, 'LineWidth', 0.5);
end

% --- 2) Scatter points
scatter(repmat(1:n_cond, n_per_cc, 1), Dmat, 15, point_color, 'filled', ...
    'MarkerFaceAlpha', 0.6, 'MarkerEdgeColor', 'none');

% --- 3) Add boxplots
switch plotType
    case 'scatBox'
        boxplot(Dmat, 'Colors', [0.2 0.2 0.2], 'Symbol', '', 'Widths', 0.4);
    case 'scatViolin'
        violinplot(Dmat)
    otherwise
        error('InvalidPlotType:%s not a valid plot type',plotType)
end

% --- 4) Beautify axes
set(gca, 'XTick', 1:n_cond, 'Box', 'off', 'TickDir', 'out');
xlabel('Condition');
xticklabels(condLabels)
ylabel('Value');
title('Pretty scatter with lines and boxplots');
xlim([0.5 n_cond+0.5]);
hold off
end
