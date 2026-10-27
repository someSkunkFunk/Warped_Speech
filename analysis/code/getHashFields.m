function flds=getHashFields(configType)
% helper function that defines "opt-in" params that should affect the hash
% registry in load/save_checkpoint
%TODO: remove the following fields from here (or from config definition 
% functions if not present in whitelists below) and re-factor
%   down-stream code (config-writing functions, analysis scripts) 
% to define them when needed:
% -------------------------------------------------------------------------
% REMOVE from PREPROCESS: 
%   stim_delay_time - not sure why I deemed this unnecessary, possibly
%   because meaningless in regIrreg because no noise addition = no extra
%   second? so experiment 
%   experiment (?) - 
%       -> does not need to be hashed, simply functions as helper for labelling
%       plots... we actually want it to be mutable without necessitating a new
%       hash/trf calculation pipeline
% -------------------------------------------------------------------------
% REMOVE from TRF-FRWD: 
%   do_lambda_optimization - can just define empty lam
%       range to imply desired behavior, then have manually-defined "best" 
%       lambda in analysis script
%   have not removed yet because it would require refactor of a bunch of
%   other code I dont currently have time to do and is not that important 
%%:IDK???
% SHOULD SUBJ REMAIN ? DO WE WANT TO POSSIBLY HAVE HASH-CONSISTENCY ACROSS
% SUBJECTS?
switch configType
    case 'preprocess'
        flds={'bpfilter','ref','fs','interpBadChans','bad_chans', ...
            'manually_selected_bad_chans','stim_delay_time', ... 
            'epoch_dur','use_triggers'};
    case 'trf'
        flds={'lam_range','tmin_ms','tmax_ms', ...
            'do_lambda_optimization','separate_conditions', ...
            'crossvalidate', 'zscore_envs', 'norm_envs', ...
            'zscore_eeg', 'use_triggers', ...
            'subsample_trfs','sep_ridge','preprocess_config', 'trf_direction'};
    case 'fEntrainment'
        flds={'fs','silenceThresh','minSilenceDur','erpWindow', 'trfLags', ...
            'lambdaGrid','preprocess_config','trf_config'};
    otherwise
        error('unknown configType "%s" add hash fields explicitly here.', ...
            configType)

end
end