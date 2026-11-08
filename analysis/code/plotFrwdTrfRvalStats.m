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

nTrialsPerCond=25; %TODO: what if we wanna plot all conditions?
nChans=128;
nCond=3;
allSubjRvals=nan(ss,nCond,nTrialsPerCond,nChans);

for ss=1:numel(subjs)
    subj=subjs(ss);
    setTrfAnalysisParams
    metadata(ss).trf_config=trf_config; metadata(ss).preprocess_config=preprocess_config;
    allSubjRvals(ss,:,:,:)=statsLoadWrapper(trf_config);
    clear subj trf_config preprocess_config
end

%% -- PLOT SUBJECT-LEVEL PRETTY SCATS (BOXPLOT)
baseCfg.configType='fwdTrfRvalStats';

for ss=1:numel(subjs)
    subj=subjs(ss);
    subjCfg_=baseCfg;
    subjCfg_.experiment=experimentsBySubjNum{ss}; % note: not assuming all subjects from same experiment
    subjCfg_=addResultsPaths(subjCfg_,subj);
    condLabels_=metadata(ss).trf_config.conditions;
    fig=prettyScat(squeeze(allSubjRvals(ss,:,:,:)),condLabels_,1);
    title(sprintf('Subj %d TRF Accuracies',subj))
    exportFiguresToFolder(subjCfg_.paths.figDir,metadata(ss),fig,0)
    clear subj condLabels_
end


%% -- PLOT ACROSS SUBJECTS PRETTY SCATS (VIOLIN)




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
function fig=prettyScat(D,condLabels,condDim)
% PRETTYSCAT scatterplot with per-sample connecting lines and boxplots OR violin plots.
%
% pretty_scat(D, cond_dim)
%
% D - data matrix of any dimensionality
% cond_dim - index of the dimension to use at the x-axis (condition/pairs)
%
% Permutes D so that cond_dim is last, then reshapes to [n_smaples x
% n_cond] and plots grey connecting lines, scatter points, and boxplots.

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
boxplot(Dmat, 'Colors', [0.2 0.2 0.2], 'Symbol', '', 'Widths', 0.4);

% --- 4) Beautify axes
set(gca, 'XTick', 1:n_cond, 'Box', 'off', 'TickDir', 'out');
xlabel('Condition');
xticklabels(condLabels)
ylabel('Value');
title('Pretty scatter with lines and boxplots');
xlim([0.5 n_cond+0.5]);
hold off
end
