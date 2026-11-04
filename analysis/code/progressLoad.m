function data=progressLoad(inputConfig)
% checkpoint_data_struct=load_checkpoint(config)
% all variables associated with config structure packaged into a structure
% is returned, unpack:
% desired_var=checkpoint_data_struct.desired_var;
    subj=inputConfig.subj;
    registryFpth=fullfile(inputConfig.paths.outputDir,'registry.json');
    if ~isfile(registryFpth)
        % note: not sure if desired behavior was for this to output an
        % empty array when registry could not be located for some
        % downstream reason, but currently it feels like it is always
        % treated as an error?
        % ^ turns out that was the original functionality in main analysis
        % script. decided a good compromise was to turn it into a warning
        % and return empty arrays.
        % error('No registry found for subject %02d\n',subj);
        warning('Could not locate registry at %s\n for subj:%d\n& configType:%s', ...
            registryFpth,subj,inputConfig.configType)
        data=[];
        return
    end
    registry=jsondecode(fileread(registryFpth));

    % normalize struct structure to get consistent hashes
    % configToHash=remove_nested_paths(config);

    % config_hash=char(upper(DataHash(configToHash)));
    hashableConfig=buildHashableConfig(inputConfig);
    % note computeConfigHash calls buildHashableConfig internally and
    % actually expects the non-hashable version (including type) as input
    configHash=computeConfigHash(inputConfig);
    configMatchIdx=find(strcmp({registry.hash},configHash));

    if isempty(configMatchIdx)
        % hashableConfig=filterConfig(config,getHashFields(config.configType));
        disp(hashableConfig)
        warning(['Existing registry contains no matching hash (%s) for\n' ...
            ' subj: %d\ninput configType:%s\nreturning empty array'], ...
            configHash,subj,inputConfig.configType)
        % 
        data=[];
        return
    else
        if length(configMatchIdx)>1
            warning(['redundant entries found... SELECTING ' ...
                'FIRST MATCHING ENTRY...'])
            configMatchIdx=configMatchIdx(1);
        end
        dataFile=registry(configMatchIdx).file;
        [dir,~,~]=fileparts(registryFpth);
        data=load(fullfile(dir,dataFile));
        fprintf(['finished loading %s checkpoint for ' ...
            'subj %02d (hash %s)\n'],inputConfig.configType,subj,configHash)
    end

end

