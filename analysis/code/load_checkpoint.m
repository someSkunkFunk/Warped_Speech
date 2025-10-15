function data=load_checkpoint(config)
    subj=config.subj;
    registry_file=fullfile(config.paths.output_dir,'registry.json');
    if ~isfile(registry_file)
        fprintf('No registry found for subject %02d\n',subj);
        data=[];
        return
    end
    registry=jsondecode(fileread(registry_file));

    % paths may differ by machine but doesn't matter for actual params
    config=rmfield(config,'paths');
    % config_str=jsonencode(config);
    % config_hash=char(upper(DataHash(config_str)));
    config_hash=char(upper(DataHash(config)));
    config_match_idx=find(strcmp({registry.hash},config_hash),1);

    if isempty(config_match_idx)
        warning('no matching file found for config below in existing registry.')
        disp(config)
        data=[];
    else
        file=registry(config_match_idx).file;
        data=load(file);
        % data=S.data;
        fprintf(['finished loading checkpoint for ' ...
            'subj %02d (hash %s)\n'],subj,config_hash)
    end
end

