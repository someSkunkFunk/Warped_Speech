% script for running update_reshash_config on multiple subjects
%% setup 
% --- NUISANCE PARAMS IN TRF_ANALYSIS PARAMS ---
script_config=[];
script_config.show_tuning_curves=true;

for subj=[2:7,9:23,96,98]
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