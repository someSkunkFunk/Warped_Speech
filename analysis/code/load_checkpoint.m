function data=load_checkpoint(config)
% checkpoint_data_struct=load_checkpoint(config)
% all variables associated with config structure packaged into a structure
% is returned, unpack:
% desired_var=checkpoint_data_struct.desired_var;
    subj=config.subj;
    registryFile=fullfile(config.paths.output_dir,'registry.json');
    if ~isfile(registryFile)
        fprintf('No registry found for subject %02d\n',subj);
        data=[];
        return
    end
    registry=jsondecode(fileread(registryFile));

    % normalize struct structure to get consistent hashes
    % configToHash=remove_nested_paths(config);
    % configToHash=filterConfig(config,getHashFields(config.configType));
    % config_hash=char(upper(DataHash(configToHash)));
    configHash=computeConfigHash(config);
    configMatchIdx=find(strcmp({registry.hash},configHash));

    if isempty(configMatchIdx)
        warning('no matching file found for config below in existing registry.')
        disp(configToHash)
        data=[];
    else
        if length(configMatchIdx)>1
            warning(['redundant entries found... SELECTING ' ...
                'FIRST MATCHING ENTRY...'])
            configMatchIdx=configMatchIdx(1);
        end
        [~,file,~]=fileparts(registry(configMatchIdx).file);
        [dir,~,~]=fileparts(registryFile);
        data=load(fullfile(dir,file));
        fprintf(['finished loading checkpoint for ' ...
            'subj %02d (hash %s)\n'],subj,configHash)
    end

end

