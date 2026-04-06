% script for running update_reshash_config on multiple subjects -- use to
% add new fields to a config and update the hashes in associated registry
%% setup 
% --- NUISANCE PARAMS IN TRF_ANALYSIS PARAMS ---
script_config=[];
script_config.show_tuning_curves=false;

for subj=[2:7,9:23,96,98]
    % initialize configs with updated fields
    trf_analysis_params;
    fprintf('**********UPDATING SUBJ %d CONFIGS***********\n', subj)
    % need to run again for both settings of separate conditions --
    % whichever is run recond will skip updating preprocess config again.
    fprintf('trf_config.separate_conditions: %d\n',trf_config.separate_conditions);
    % assumes these exist already and just need to add fields not
    % previously saved
    disp('updating preprocess config')
    update_rehash_config(preprocess_config)
    disp('done.')
    disp('updating trf config')
    update_rehash_config(trf_config)
    disp('done.')
    clearvars -except script_config subj
end