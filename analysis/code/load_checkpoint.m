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
    config_match_idx=find(strcmp({registry.hash},config_hash));

    if isempty(config_match_idx)
        warning('no matching file found for config below in existing registry.')
        disp(config)
        data=[];
    elseif length(config_match_idx)>1
        %TODO: we accidentally saved redundant entires.... need to remedy
        %this by consolidating files with the same hash - and pruning the
        %registry. I _THINK_ we probably can just use save with append flag
        % to consolidate the variables in a single file... but should
        % VERIFY this is true before moving forward with it
        error('fix this shit')
    else
        file=registry(config_match_idx).file;
        data=load(file);
        % data=S.data;
        fprintf(['finished loading checkpoint for ' ...
            'subj %02d (hash %s)\n'],subj,config_hash)
    end
end

