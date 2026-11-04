% plotBkwdTrfResults
% PLOT BACKWARDS PREDICTION ACCURACIES
% USAGE
% set subj: single number or array with multiple subj nums
% trfAnalysisPipeline
%   > will load or compute results
% works both for 1-subject or multi-subjects stats_obs_bkwd

% stats_obs_bkwd(ss).r, .err -> nTrials x nConditions

%% CONFIG USER SETTINGS

exportFigs=false; % true export each figure and close it right after
% outDir=_ probably needs to be done per subject

nSubj=numel(stats_obs_bkwd);
nCond=size(stats_obs_bkwd(1).r, 2);
condLabels=arrayfun(@(c) sprintf('Cond %d', c), 1:nCond, 'UniformOutput',false);
%% PER-SUBJECT SUMMARIES (always compute)
r_subjMean=nan(nSubj, nCond);
r_subjSEM=nan(nSubj,nCond);
err_subjMean=nan(nSubj, nCond);
err_subjSEM=nan(nSubj,nCond);

for ss=1:nSubj
    r_subjMean(ss,:)=mean(stats_obs_bkwd(ss).r,1);
    % r_subjSEM(ss,:)= % TODO: SEM over trials (what if nTR ==1?)
end

%% One figure per subject
for ss=1:nSubj
    subjNum_=subj(ss);
    fig=figure('Color','w','Name', sprintf('Subject %d', subjNum_, 'NumberTitle','off'));

    ax1=subplot(1,2,1); hold(ax1, 'on');
    plotTrialsWithMean(ax1, stats_obs_bkwd(ss).r, condLabels);
    ylabel(ax1, 'Reconstruction Accuracy (r)');
    title(ax1,'r');

    ax2=subplot(1,2,2); hold(ax2,'on');
    %TODO: same helper, but for err (do we need this?)
    title('err');

    sgtitle(sprintf('Subject %d (trials = dots, black = mean \\pm SEM',subjNum_))
    if exportFigs
        %TODO: Call export function
        metadata_=buildConfigPathsGeneric(subjNum,{})
        exportFiguresToFolder(fig,{},fig,1)

        clear figDir_
    end
    clear subjNum_
end