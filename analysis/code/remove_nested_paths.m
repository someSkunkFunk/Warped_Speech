function S = remove_nested_paths(S)
% REMOVE_NESTED_PATHS Recursively Remove all 'paths' fields at any depth.
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
                val{ii} = remove_nested_paths(val{ii});
            end

            % Write back safely
            [S.(fn{ff})] = val{:};
        end

    % ---- CELL ARRAY ----
    elseif iscell(S)
        for ii = 1:numel(S)
            S{ii} = remove_nested_paths(S{ii});
        end
    end

    % ---- EVERYTHING ELSE: DO NOTHING ----
end