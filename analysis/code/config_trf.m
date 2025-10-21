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
    'zscore_eeg',true,...
    'experiment',[], ...
    'conditions',[],...
    'train_params',struct('tmin_ms', -500, ...
    'tmax_ms',800, ...
    'best_lam',0) ...
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
    trf_config.experiment=preprocess_config.experiment;
end


if trf_config.separate_conditions
    switch trf_config.experiment
        case 'fast-slow'
            trf_config.conditions={'fast','original','slow'};
        case 'reg-irreg'
            trf_config.conditions={'reg','original','irreg'};
        otherwise
            warning('experiment "%s" undefined.',experiment)
    end
else
    trf_config.conditions={'all conditions'};
end

%%%%%%%%%%%%%%%%%%%%%% PATHS TO IGNORE IN REGISTRY %%%%%%%%%%%%%%%%%%%%%%%%
if trf_config.subj>=90
    %TODO: avoid hardcoding 90 somehow...?
    trf_config.paths.envelopesFile=sprintf('%s/stimuli/wrinkle/RegIrregPilotEnvelopes%dhz.mat', boxdir_mine, ...
        preprocess_config.fs);
else
    trf_config.paths.envelopesFile=sprintf('%s/stimuli/wrinkle/Envelopes%dhz.mat', boxdir_mine, ...
        preprocess_config.fs);
end

trf_config.paths.output_dir=sprintf('%s/analysis/trf_models/%02d',boxdir_mine,trf_config.subj);

trf_config=orderfields(trf_config);
disp('voila trf_config:')
disp(trf_config)
% guess won't need this since function only returns trf_config
% clearvars -except trf_config 

end


