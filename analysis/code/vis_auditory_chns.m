% helper for visualizing channels with high prediction accuracy
% uses prediction accuracies from condition-agnostic TRF fitting 

%% --- GENERAL SETTINGS ---
N=12; % number of top-performing electrodes to keep

% for each subject
subjs=[2];
script_config=struct('show_tuning_curves',false);

for subj=subjs
%% load stats_obs from cross validation
trf_analysis_params
trf_checkpoint=load_checkpoint(trf_config);
stats_obs=trf_checkpoint.stats_obs;
clear trf_checkpoint
max_lam_idx=get_best_lambda(stats_obs); % [1 x lambdas]
best_rs=squeeze(stats_obs.r(:,max_lam_idx,:)); clear stats_obs % [trials x chns]

%% for each crossvalidation step, rank electrodes by prediction accuracy 
[~, Isort]=sort(best_rs,2);
%% collect top 12 electrodes with highest prediction accuracy 
% top channels from 67/75 of trials (~89%)


%% visualize topo marking top channels

%% plot individual channel prediction accuracies 
% as scatter with x: chn num y: prediction accuracy
% color in red those selected in above procedure

end

%TODO: how to visualize consistency across subjects?

function max_lam_idx=get_best_lambda(stats_obs)
rs_all=stats_obs.r; % [trials x lambdas x chns]
% average across electrodes & chns
rs_vs_lam=mean(rs_all,[1,3]); %[1 x lambdas]
[~,max_lam_idx]=max(rs_vs_lam);
end