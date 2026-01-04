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
    % make sure lam_range is row vector... (maybe should have a more
    % general solution to make all row vectors into column vectors for
    % consistency)
    if isfield(config,'lam_range')
        if isrow(config.lam_range)
            % transpose
            config.lam_range=config.lam_range.';
        end
    end

    % config_str=jsonencode(config);
    % config_hash=char(upper(DataHash(config_str)));
    % nromalize structures to save vectors as column vectors
    function X = normalize_row_vectors(X)
    %NORMALIZEROWVECTORS
    % Recursively converts row vectors (1xN) to column vectors (Nx1)
    % inside structs, struct arrays, and cell arrays.
    
        % ---- Numeric or logical ---------------------------------------------
        if isnumeric(X) || islogical(X)
            if isvector(X) && size(X,1) == 1 && size(X,2) > 1
                X = X.';   % force column
            end
            return
        end
    
        % ---- Struct ----------------------------------------------------------
        if isstruct(X)
            for k = 1:numel(X)
                f = fieldnames(X(k));
                for i = 1:numel(f)
                    X(k).(f{i}) = normalize_row_vectors(X(k).(f{i}));
                end
            end
            return
        end
    
        % ---- Cell ------------------------------------------------------------
        if iscell(X)
            for i = 1:numel(X)
                X{i} = normalize_row_vectors(X{i});
            end
            return
        end
    
        % ---- Everything else -------------------------------------------------
        % char, string, function_handle, objects, etc.
        % Leave untouched
    end
    config=normalize_row_vectors(config);
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

