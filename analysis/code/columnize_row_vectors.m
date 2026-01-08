% NOTE: this function can probably go in a more general-utility
% directory...
function X = columnize_row_vectors(X)
    %NORMALIZEROWVECTORS
    % Recursively converts row vectors (1xN) to column vectors (Nx1)
    % inside structs, struct arrays, and cell arrays.
    
        % ---- Numeric or logical ---------------------------------------------
        if isnumeric(X) || islogical(X)
            if isrow(X)
                X = X.';   % force column
            end
            return
        end
    
        % ---- Struct ----------------------------------------------------------
        if isstruct(X)
            for k = 1:numel(X)
                f = fieldnames(X(k));
                for i = 1:numel(f)
                    X(k).(f{i}) = columnize_row_vectors(X(k).(f{i}));
                end
            end
            return
        end
    
        % ---- Cell ------------------------------------------------------------
        if iscell(X)
            for i = 1:numel(X)
                X{i} = columnize_row_vectors(X{i});
            end
            return
        end
    
        % ---- Everything else -------------------------------------------------
        % char, string, function_handle, objects, etc.
        % Leave untouched
    end