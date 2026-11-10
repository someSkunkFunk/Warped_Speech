dataDir='C:\Users\ninet\Box\AndreJinSharedFolder\noise_ceiling_results';

D=dir(fullfile(dataDir,'*.json'));
nCond=3;
nFiles=numel(D);
nSubjs=nFiles/nCond; %TODO: CHECK IF DIVISIBLY BY nCOND
assert(mod(nFiles,nSubjs)==0)
allSubjsData=cell(nSubjs,nCond);

baseCfg.configType='plotExplainableVariaince';
baseCfg.condLabels={'reg', 'original', 'irreg'};

%% get subjs
subjs=nan(nSubjs,1);
for ff=1:nFiles
    fileName_=fullfile(D(ff).folder,D(ff).name);
    [subjNum_,~]=getSubjInfo(fileName_);
    % subjMatch_=regexp(fileName_,'subject~(\d+)','tokens');
    % subjNum_=str2double(subjMatch_{1}{1});
    if ~any(subjNum_==subjs)
        subjs(find(isnan(subjs),1))=subjNum_;
    end
    clear subjMatch_ subjNum_ fileName_
end
%%
for ff=1:nFiles
    fileName_=fullfile(D(ff).folder,D(ff).name);
    % extract condition number and subject numbers using regex
    % condMatch_=regexp(fileName_,'condition~(\d+)','tokens');
    % subjMatch_=regexp(fileName_,'subject~(\d+)','tokens');
    [subjNum_, condNum_]=getSubjInfo(fileName_);
    % condNum=str2double(condMatch_{1}{1});
    allSubjsData{subjs==subjNum_,condNum_}=jsondecode(fileread(fileName_));
    clear condNum_ subjNum_ fileName_
end

%% PLOTR


% aggregate individual subject noise ceiling
subjNoiseCeilings=nan(nSubjs,nCond);
for ss=1:nSubjs
    % get noiseCeiling across conditions
    subjData_=vertcat(allSubjsData{ss,:});
    subjNoiseCeilings(ss,:)=[subjData_.center];

end

fig=prettyScat(subjNoiseCeilings,baseCfg.condLabels,2,'scatViolin');

%%
function [subjNum, condNum]=getSubjInfo(fileName)
condMatch=regexp(fileName,'condition~(\d+)','tokens');
subjMatch=regexp(fileName,'subject~(\d+)','tokens');
condNum=str2double(condMatch{1}{1});
subjNum=str2double(subjMatch{1}{1});

end

function fig=prettyScat(D,condLabels,condDim,plotType)
% PRETTYSCAT scatterplot with per-sample connecting lines and boxplots OR violin plots.
% TODO: SEPARATE THIS FUNCTION TO IS OWN .m FILE TO AVOID CONFLICTS ACROSS
% SCRIPTS?
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
