function trfConfig=configTrf(trfConfig,preprocess_config)
%note: is it better to have this depend on input preprocess_config or have
%it be standalone such that we can call it on it's own, in which case
%should run config_preprocess here if empty....?
% sep_ridge: true -> optimize lambda separately for each condition
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
    'use_triggers',[], ...
    'subsample_trfs',false, ...
    'sep_ridge',false, ...
    'trf_direction',1 ...
    );
if isempty(preprocess_config)
    warning('empty preprocess_config given...')
    disp('initializing one with defaults to get trf_config paths...')
    disp('but this is probably not what you want for analysis')
    preprocess_config=config_preprocess([]);
end

fields=fieldnames(defaults);
for ff=1:numel(fields)
    if ~isfield(trfConfig,fields{ff})||isempty(trfConfig.(fields{ff}))
        if isfield(preprocess_config,fields{ff})
            % copy redundant fields
            trfConfig.(fields{ff})=preprocess_config.(fields{ff});
        else
            trfConfig.(fields{ff})=defaults.(fields{ff});
        end
    end
end




if trfConfig.separate_conditions
    switch trfConfig.experiment
        case 'fastSlow'
            trfConfig.conditions={'fast','original','slow'};
        case 'regIrreg'
            trfConfig.conditions={'reg','original','irreg'};
        otherwise
            warning('experiment "%s" undefined.',experiment)
    end
else
    trfConfig.conditions={'all conditions'};
end
% remove paths from preprocess config to avoid hash errors in different
% systems
% preprocess_config=rmfield(preprocess_config,'paths');
% preprocess_config=filterConfig(preprocess_config, ...
%     getHashFields(preprocess_config.config_type));
preprocess_config=buildHashableConfig(preprocess_config);
trfConfig.preprocess_config=preprocess_config;

%%%%%%%%%%%%%%%%%%%%%% PATHS TO IGNORE IN REGISTRY %%%%%%%%%%%%%%%%%%%%%%%%
if trfConfig.subj>=90
    %TODO: avoid hardcoding 90 somehow...?
    trfConfig.paths.envelopesFile=sprintf('%s/stimuli/wrinkle/RegIrregPilotEnvelopes%dhz.mat', boxdir_mine, ...
        preprocess_config.fs);
elseif trfConfig.subj>23&&trfConfig.subj<90
    % 23 left out intentionally because stimuli were different
    trfConfig.paths.envelopesFile=sprintf('%s/stimuli/wrinkle/RegIrregEnvelopes%dhz.mat', boxdir_mine, ...
        preprocess_config.fs);
elseif trfConfig.subj<23
    trfConfig.paths.envelopesFile=sprintf('%s/stimuli/wrinkle/fastSlowEnvelopes%dhz.mat', boxdir_mine, ...
        preprocess_config.fs);
end

trfConfig.paths.output_dir=sprintf('%s/analysis/%s/trf_models/%02d', ...
    boxdir_mine,trfConfig.experiment,trfConfig.subj);
trfConfig.configType='trf';
% column-major order for consistency
trfConfig=orderfields(trfConfig);
trfConfig=columnize_row_vectors(trfConfig);
disp('voila trf_config:')
disp(trfConfig)

end


