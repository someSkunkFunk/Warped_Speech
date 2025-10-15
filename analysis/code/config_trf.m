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
    'tmin_ms',0, ...
    'tmax_ms',400, ...
    'do_lambda_optimization',false,...
    'separate_conditions',false,...
    'crossvalidate',false,...
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
trf_config.paths.envelopesFile=sprintf('%s/stimuli/wrinkle/WrinkleEnvelopes%dhz.mat', boxdir_mine, ...
    preprocess_config.fs);


%% new format vars to replace single nuldistribution file output
% trf_config.paths.model_metric_path=sprintf('%s/analysis/metrics/warped_speech_s%0.2d.mat', ...
%     boxdir_mine,trf_config.subj);
% trf_config.paths.trf_model_path=sprintf('%s/analysis/models/warped_speech_s%0.2d.mat', ...
%     boxdir_mine,trf_config.subj);

trf_config.paths.output_dir=sprintf('%s/analysis/trf_models/%02d',boxdir_mine,trf_config.subj);

trf_config=orderfields(trf_config);
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

