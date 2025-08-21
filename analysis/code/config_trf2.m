function trf_config=config_trf2(subj,do_lambda_optimization,preprocess_config)
%% %trf params config file
global user_profile
global boxdir_lab
global boxdir_mine
% fprintf(['note that variable called separate_conditions has been ' ...
%     'replaced with do_lambda_optimization and downstream code ' ...
%     'might need to be updated to reflect proper functionality if it uses' ...
%     'the old variable....\n'])
% although ideally, packaging the directories and logic used to located 
% them into this function is the point of making this function to begin
% with so that should not be an issue...
%% stuff that might change depending on context
% optimization params
%range over which we optimize lambda with all conditions combined
lam_range=10.^(-3:8);
cv_tmin_ms=0;
cv_tmax_ms=400;

%model params
tmin_ms=-500;
tmax_ms=800;
%% stuff that doesn't generally need to change
% TODO: currently, we only do cross-validation when we optimize lambda
% (using cross-condition TRFs), we will definitely be re-structuring
% project directories in future so that all model results end up on the
% same file, at which point we probably want to make it possible for
% condition-specific TRFs to be optimized and/or cross validated
% independently of the condition-agnostic TRFs

if do_lambda_optimization
    conditions_dir='all_conditions/';
    cvdir='cv/';
    %for new output file format - just store as variables within the same
    %file instead of dividing into separate dirs
    separate_conditions=false;
    crossvalidated=true;

else
    conditions_dir='sep_conditions/';
    cvdir='';
    
    separate_conditions=true;
    crossvalidated=false;
end

zscore_envs=false;
norm_envs=true;
zscore_eeg=true;

nulldistribution_file=sprintf('%s%s%snulldistribution_s%0.2d.mat', ...
    preprocess_config.matfolder,conditions_dir,cvdir,subj);
envelopesFile=sprintf('%s/stimuli/WrinkleEnvelopes%dhz.mat', boxdir_mine, ...
    preprocess_config.fs);


%% new format vars to replace single nuldistribution file output
model_metric_path=sprintf('%s/analysis_interpBadChans/metrics/warped_speech_s%0.2d.mat', ...
    boxdir_mine,subj);
trf_model_path=sprintf('%s/analysis_interpBadChans/models/warped_speech_s%0.2d.mat', ...
    boxdir_mine,subj);

model_metric_dir=fileparts(model_metric_path);
if ~exist(model_metric_dir,'dir')
    fprintf('%s DNE - making dir ...\n',model_metric_dir)
    mkdir(preprocess_config.preprocessed_eeg_path)
end

trf_model_dir=fileparts(trf_model_path);
if ~exist(trf_model_dir,'dir')
    fprintf('%s DNE - making dir ...\n',trf_model_dir)
    mkdir(preprocess_config.preprocessed_eeg_path)
end


vars=whos;
trf_config=struct();
for nn=1:numel(vars)
    trf_config.(vars(nn).name)=eval(vars(nn).name);
end

fprintf('voila trf_config:\n')
disp(trf_config)
% guess won't need this since function only returns trf_config
% clearvars -except trf_config 

end
