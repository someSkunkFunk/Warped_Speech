% helper for visualizing channels with high prediction accuracy
% uses prediction accuracies from condition-agnostic TRF fitting 

%% --- GENERAL SETTINGS ---
N=12; % number of top-performing electrodes to keep

% for each subject
subjs=[2];
script_config=struct('show_tuning_curves',false);
chanlocs=load(loc_file);
chanlocs=chanlocs.chanlocs;
nchns=128;
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
% for each cv step, get top N channels


top_nchns_list=unique(Isort(:,1:N));
nchn_ranks=nan(size(top_nchns_list));
% count how many CV steps they occur in
for te=1:length(top_nchns_list)
    nchn_ranks(te)=sum(Isort(:,1:N)==top_nchns_list(te),'all');
end
% note: approach below doesn't give any rankings that would be above 89%
% threshold, so either I misinterpreted what they said they did or their
% more extensive pre-processing/data quality boosted their prediction
% accuracies...


% workaround: just pick top 12 and see if they are at least approximately
% right
[~,Isort_ranks]=sort(nchn_ranks);
vis_chns=top_nchns_list(Isort_ranks);
vis_chns=vis_chns(1:N);

%% visualize topo marking top channels
vis_chns=[54 55 56 61 62 63 106 107 108 115 116 117];
figure("Name",sprintf("subj %d top %d channels",subj,N))
topoplot(zeros(1,nchns),chanlocs, ...
    'style','fill', ...
    'electrodes','off', ...
    'emarker2',{vis_chns,'o','red',8,1})
%% plot individual channel prediction accuracies 
% as scatter with x: chn num y: prediction accuracy
% color in red those selected in above procedure

end

%TODO: how to visualize consistency across subjects?

function max_lam_idx=get_best_lambda(stats_obs)
rs_all=stats_obs.r; % [trials x lambdas x chns]
% average across trials & chns
rs_vs_lam=mean(rs_all,[1,3]); %[1 x lambdas]
[~,max_lam_idx]=max(rs_vs_lam);
end