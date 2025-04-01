function save_checkpoint(output_config,output_data)
% helper to save checkpoints at different points of analysis
%TODO: determine if config validation needs to be taken out of
%load_checkpoint to be accessed here, for now assuming that this is only
%going to run when load_checkpoint not an option hence either matching
%config does not exist OR it does but the checkpoint variable to be saved is not
%included in the file 
%NOTE: i think the facts above imply there may be organizational 
% issues if we don't run analysis_script with do_nulltest=true;
% script already saved to file
config_name=inputname(1);
if ~contains(config_name,'config')
    error('put config file first')
end
data_name=inputname(2);


%TODO: for each statement in analysis_script below, in order to be replaced
%by this function, need the create a "mapping" from config variable/data
%names to the appropriate path-field in the passed config


% save(preprocess_config.preprocessed_eeg_path,'preprocessed_eeg','preprocess_config');
% save(trf_config.model_metric_path,'stats_obs','trf_config');
% fprintf('append-saving stats_null to %s...\n',trf_config.model_metric_path)
%save(trf_config.model_metric_path,'stats_null','-append')
% save(trf_config.trf_model_path,'model','trf_config');

% map variable to output file save path using structure array
%NOTE: any value to tracking appropriate config varname as well?
%NOTE: using a table might be better/less confusing.... but not sure?
config_output_paths.stats_obs='model_metric_path';
config_output_paths.stats_null='model_metric_path';
config_output_paths.preprocessed_eeg='preprocessed_eeg_path';
config_output_paths.model='trf_model_path';

output_path=config_output_paths.(data_name);
output_stage=0;
if exist(output_path,"file")
    output_stage=output_stage+1;
end
if output_stage>0&&ismember(data_name,who("-file",output_path))
    % file contains output var, need to append current result to it
    output_stage=output_stage+2;
else
    % file does not contain output var
    output_stage=output_stage+1;
end
%TODO: verify each save case!!!
switch output_stage
    case 0
        %% initialize nonexisting file 
        % use struct flag to save each field of dummy struct as individual var
        % use a dummy struct array to save variables with correct name
        output_struct=struct(config_name,output_config,data_name,output_data);
        save(output_path,"-struct","output_struct")
    case 1
        %% append-save new vars to existing file
        % note this assumes config validation not needed when running this
        % function....
        output_struct=struct(data_name,output_data);
        save(output_path,"-struct","output_struct","-append")

    case 2
        %% append current data var + config to end of existing data var
        % vars, then re-save file
        % output_struct=struct(config_name,output_config,data_name,output_data);
        existing_file_data=load(output_path);
        % output_struct=struct(config_name,output_config,data_name,output_data);
        % concatenate existing data and output struct data accounting for
        % possible differences in number of dimensions encoded in existing
        % data vs output data
        m_conditions_out=size(output_data,2);
        m_conditions_file=size(existing_file_data.(data_name),2);

        if m_conditions_out==m_conditions_file
            % can just cat
            output_struct=struct(config_name, ...
                cat(1,existing_file_data.(config_name),output_config), ...
                data_name,cat(1,existing_file_data.(data_name),output_data));
        else
            % assign each condition struct one at a time to expand dim of
            % existing struct array

            %note that config doesnt need new dims
            expanded_data=existing_file_data.(data_name);
            n_configs_saved=size(expanded_data,1);
            for cc=1:m_conditions_out
                expanded_data(n_configs_saved+1,cc)=output_data(1,cc);
            end
            output_struct=struct(config_name, ...
                cat(1,existing_file_data.(config_name),output_config), ...
                data_name,expanded_data);
           
        end
        
        
        save(output_path,"-struct","output_struct","-append")

end
end