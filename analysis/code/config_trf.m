function trf_config=config_trf(trf_config,preprocess_config)
%note: is it better to have this depend on input preprocess_config or have
%it be standalone such that we can call it on it's own, in which case
%should run config_preprocess here if empty....?
%% %trf params config file
global user_profile
global boxdir_lab
global boxdir_mine
defaults=struct( ...
    'subj',[],...
    'lam_range',10.^(-3:8), ...
    'cv_tmin_ms',0, ...
    'cv_tmax_ms',400, ...
    'tmin_ms',-500, ...
    'tmax_ms',800, ...
    'do_lambda_optimization',false,...
    'separate_conditions',false,...
    'crossvalidated',false,...
    'zscore_envs',false, ...
    'norm_envs',true,...
    'zscore_eeg',true...
    );
fields=fieldnames(defaults);
for ff=1:numel(fields)
    if ~isfield(trf_config,fields{ff})||isempty(trf_config.(fields{ff}))
        trf_config.(fields{ff})=defaults.(fields{ff});
    end
end

if isempty(preprocess_config)
    warning('empty preprocess_config given...')
    disp('initializing one with defaults to get trf_config...')
    disp('avoid this when doing actual analysis')
    preprocess_config=config_preprocess([]);
else
    trf_config.subj=preprocess_config.subj;
end
%% stuff that might change depending on context
% optimization params
%range over which we optimize lambda with all conditions combined
% lam_range=10.^(-3:8);
% cv_tmin_ms=0;
% cv_tmax_ms=400;

%model params
% tmin_ms=-500;
% tmax_ms=800;
%% stuff that doesn't generally need to change
%NOTE: consider making lambda optimization independent separate
%conditions.... or making them dependent in outer script rather than here
% if trf_config.do_lambda_optimization
%     conditions_dir='all_conditions/';
%     cvdir='cv/';
%     %for new output file format - just store as variables within the same
%     %file instead of dividing into separate dirs
%     trf_config.separate_conditions=false;
%     crossvalidated=true;
% 
% else
%     conditions_dir='sep_conditions/';
%     cvdir='';
% 
%     separate_conditions=true;
%     crossvalidated=false;
% end

% zscore_envs=false;
% norm_envs=true;
% zscore_eeg=true;

% nulldistribution_file=sprintf('%s%s%snulldistribution_s%0.2d.mat', ...
%     preprocess_config.matfolder,conditions_dir,cvdir,subj);
trf_config.envelopesFile=sprintf('%s/stimuli/WrinkleEnvelopes%dhz.mat', boxdir_mine, ...
    preprocess_config.fs);


%% new format vars to replace single nuldistribution file output
trf_config.paths.model_metric_path=sprintf('%s/analysis/metrics/warped_speech_s%0.2d.mat', ...
    boxdir_mine,trf_config.subj);
trf_config.paths.trf_model_path=sprintf('%s/analysis/models/warped_speech_s%0.2d.mat', ...
    boxdir_mine,trf_config.subj);

trf_config.paths.model_metric_dir=fileparts(trf_config.paths.model_metric_path);

% if ~exist(trf_config.paths.model_metric_dir,'dir')
%     fprintf('%s DNE - making dir ...\n',trf_config.paths.model_metric_dir)
%     mkdir(trf_config.paths.model_metric_dir)
% end

trf_config.paths.trf_model_dir=fileparts(trf_config.paths.trf_model_path);
% if ~exist(trf_config.trf_model_dir,'dir')
%     fprintf('%s DNE - making dir ...\n',trf_config.paths.trf_model_dir)
%     mkdir(trf_config.paths.trf_model_dir)
% end


% vars=whos;
% trf_config=struct();
% for nn=1:numel(vars)
%     trf_config.(vars(nn).name)=eval(vars(nn).name);
% end

disp('voila trf_config:')
disp(trf_config)
% guess won't need this since function only returns trf_config
% clearvars -except trf_config 

end
%% BANISHED REALM
% if ~isscalar(lam_range)
%     cvdir='cv/';
% else
%     cvdir='';
% end

