function data=progressLoad(config)
% checkpoint_data_struct=load_checkpoint(config)
% all variables associated with config structure packaged into a structure
% is returned, unpack:
% desired_var=checkpoint_data_struct.desired_var;
    subj=config.subj;
    registryFpth=fullfile(config.paths.output_dir,'registry.json');
    if ~isfile(registryFpth)
        fprintf('No registry found for subject %02d\n',subj);
        data=[];
        return
    end
    registry=jsondecode(fileread(registryFpth));

    % normalize struct structure to get consistent hashes
    % configToHash=remove_nested_paths(config);

    % config_hash=char(upper(DataHash(configToHash)));

    configHash=computeConfigHash(config);
    configMatchIdx=find(strcmp({registry.hash},configHash));

    if isempty(configMatchIdx)
        % hashableConfig=filterConfig(config,getHashFields(config.configType));
        hashableConfig=buildHashableConfig(config);
        warning('no matching file found for config below in existing registry.')
        disp(hashableConfig)
        data=[];
    else
        if length(configMatchIdx)>1
            warning(['redundant entries found... SELECTING ' ...
                'FIRST MATCHING ENTRY...'])
            configMatchIdx=configMatchIdx(1);
        end
        dataFile=registry(configMatchIdx).file;
        [dir,~,~]=fileparts(registryFpth);
        data=load(fullfile(dir,dataFile));
        fprintf(['finished loading checkpoint for ' ...
            'subj %02d (hash %s)\n'],subj,configHash)
    end

end

