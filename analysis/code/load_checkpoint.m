function data=load_checkpoint(config)
% checkpoint_data_struct=load_checkpoint(config)
% all variables associated with config structure packaged into a structure
% is returned, unpack:
% desired_var=checkpoint_data_struct.desired_var;
    subj=config.subj;
    registry_file=fullfile(config.paths.output_dir,'registry.json');
    if ~isfile(registry_file)
        fprintf('No registry found for subject %02d\n',subj);
        data=[];
        return
    end
    registry=jsondecode(fileread(registry_file));

    % normalize struct structure to get consistent hashes
    % configToHash=remove_nested_paths(config);
    configToHash=filterConfig(config,getHashFields(config.configType));
    config_hash=char(upper(DataHash(configToHash)));
    config_match_idx=find(strcmp({registry.hash},config_hash));

    if isempty(config_match_idx)
        warning('no matching file found for config below in existing registry.')
        disp(configToHash)
        data=[];
    else
        if length(config_match_idx)>1
            warning(['redundant entries found... SELECTING ' ...
                'FIRST MATCHING ENTRY...'])
            config_match_idx=config_match_idx(1);
        end
        [~,file,~]=fileparts(registry(config_match_idx).file);
        [dir,~,~]=fileparts(registry_file);
        data=load(fullfile(dir,file));
        fprintf(['finished loading checkpoint for ' ...
            'subj %02d (hash %s)\n'],subj,config_hash)
    end

end

