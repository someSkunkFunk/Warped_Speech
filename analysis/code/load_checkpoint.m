function data=load_checkpoint(config)
    subj=config.subj;
    registry_file=fullfile(config.paths.output_dir,'registry.json');
    if ~isfile(registry_file)
        fprintf('No registry found for subject %02d\n',subj);
        data=[];
        return
    end
    registry=jsondecode(fileread(registry_file));

    % normalize struct structure to get consistent hashes
    config=remove_nested_paths_recursive(config);
    % config=columnize_row_vectors(config); % this should be done in
    % config_trf

    config_hash=char(upper(DataHash(config)));
    config_match_idx=find(strcmp({registry.hash},config_hash));

    if isempty(config_match_idx)
        warning('no matching file found for config below in existing registry.')
        disp(config)
        data=[];
    elseif length(config_match_idx)>1
        error('redundant entries found and need to be dissolved...')
    else
        file=registry(config_match_idx).file;
        data=load(file);
        fprintf(['finished loading checkpoint for ' ...
            'subj %02d (hash %s)\n'],subj,config_hash)
    end

    function S = remove_nested_paths_recursive(S)
% REMOVE_NESTED_PATHS_RECURSIVE Remove all 'paths' fields at any depth.
%
% Safe for struct arrays, nested structs, and cell arrays.

    % ---- STRUCT OR STRUCT ARRAY ----
    if isstruct(S)

        % Remove 'paths' field from entire struct array at once
        if isfield(S, 'paths')
            S = rmfield(S, 'paths');
        end

        % Recurse into remaining fields
        fn = fieldnames(S);
        for ff = 1:numel(fn)
            val = {S.(fn{ff})};

            % Recurse elementwise if needed
            for ii = 1:numel(val)
                val{ii} = remove_nested_paths_recursive(val{ii});
            end

            % Write back safely
            [S.(fn{ff})] = val{:};
        end

    % ---- CELL ARRAY ----
    elseif iscell(S)
        for ii = 1:numel(S)
            S{ii} = remove_nested_paths_recursive(S{ii});
        end
    end

    % ---- EVERYTHING ELSE: DO NOTHING ----
end
end

