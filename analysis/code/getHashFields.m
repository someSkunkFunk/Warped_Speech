function flds=getHashFields(configType)
% helper function that defines "opt-in" params that should affect the hash
% registry in load/save_checkpoint
%TODO: remove the following fields from here (or from config definition 
% functions if not present in whitelists below) and re-factor
%   down-stream code (config-writing functions, analysis scripts) 
% to define them when needed:
% -------------------------------------------------------------------------
% REMOVE from PREPROCESS: 
%   nchan - redundant from other info and only used in preprocess
%   stim_delay_time
%   experiment (?) - does it need to be hashed or can it just be extracted from subj
%       num and be defined where needed in script? did we include it simply to
%       avoid having to rewrite basic label strings in each analysis script?
%       -> does not need to be hashed, simply functions as helper for labelling
%       plots... we actually want it to be mutable without necessitating a new
%       hash/trf calculation pipeline
% -------------------------------------------------------------------------
% REMOVE from TRF-FRWD: 
%   do_lambda_optimization - can just define empty lam
%       range to imply desired behavior, then have manually-defined "best" 
%       lambda in analysis script
%   experiment - same logic as above, also apply to anything that can be
%   recovered from preprocess config since trf_config already should have
%   it
%   conditions - just condition labels for plots? if so, don't need here

switch configType
    case 'preprocess'
        flds={'bpfilter','ref','fs','interpBadChans','bad_chans', ...
            'manually_selected_bad_chans','stim_delay_time', ... 
            'epoch_dur','experiment','use_triggers'};
    case 'trf'
        flds={'lam_range','tmin_ms','tmax_ms', ...
            'do_lambda_optimization','separate_conditions', ...
            'crossvalidate', 'zscore_envs', 'norm_envs', ...
            'zscore_eeg', 'experiment', 'conditions', 'use_triggers', ...
            'subsample_trfs','sep_ridge','preprocess_config', 'trf_direction'};
    otherwise
        error('unknown configType "%s" add hash fields explicitly here.', ...
            configType)

end
end