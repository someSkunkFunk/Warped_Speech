function save_checkpoint(data_config,save_data)
config_name=inputname(1);
data_name=inputname(2);


%TODO: for each statement in analysis_script below, in order to be replaced
%by this function, need the create a "mapping" from config variable/data
%names to the appropriate path-field in the passed config


% save(preprocess_config.preprocessed_eeg_path,'preprocessed_eeg','preprocess_config');
% save(trf_config.model_metric_path,'stats_obs','trf_config');
% fprintf('append-saving stats_null to %s...\n',trf_config.model_metric_path)
%save(trf_config.model_metric_path,'stats_null','-append')
% save(trf_config.trf_model_path,'model','trf_config');
end