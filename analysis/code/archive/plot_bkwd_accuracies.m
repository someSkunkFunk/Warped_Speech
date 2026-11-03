% plot backward reconstruction accuracies
global boxdir_mine
bkwd_dir=sprintf('%s/analysis/trf_models/BKWD MODELS',boxdir_mine);

subjs=[23:29];
n_trials=75;
% initialize array to store trial-level r-vals
Rs=nan(length(subjs),n_trials);


for ss=1:length(subjs)
    subj=subjs(ss);
    % navigate to subject folder
    subj_dir=sprintf('%s/%02d',bkwd_dir,subj);
    % grab the mat file
    D=dir([subj_dir,'/*.mat']);
    % add conditions + predicted correlations
    load([subj_dir,'/',D.name],'stats_obs_bkwd');
    Rs(ss,:)=stats_obs_bkwd.r;
end