function windows = validate_input()
%VALIDATE_INPUT  Validate time window input.
%   WINDOWS = VALIDATE_INPUT(WINDOWS) checks that WINDOWS is either:
%     (1) a row vector with an even number of elements, representing
%         [onset1 offset1 onset2 offset2 ...], or
%     (2) a N-by-2 array, where each row is [onset offset;].
%   Additional checks:
%     - Each onset must be strictly less than its corresponding offset.
%     - Windows must not overlap.
%   If WINDOWS is invalid, the user will be prompted to re-enter input
%   until a valid set of windows (or [] for none) is given.
    prompt=['Enter (array of) time window(s) enclosing peak(s)' ...
        '\nto remove (empty array if none):\n'];
    windows = input(prompt);
    while ~is_valid(windows)
        disp('Invalid input. Please enter windows again.');
        disp('Valid formats:');
        disp(' - Row vector with even length: [on1 off1 on2 off2 ...]');
        disp(' - N-by-2 array: [on1 off1 ...; on2 off2 ...]');
        disp(' - Empty array [] is also valid.');
        windows = input(prompt);
    end
    
    % Normalize format: always return 2-by-N
    if isempty(windows)
        return;
    elseif isvector(windows)
        windows = reshape(windows, 2, []);
    end

    function ok = is_valid(w)
        ok = false;
        
        if isempty(w)
            ok = true;
            return;
        end
        
        % Case 1: row vector with even number of elements
        if isrow(w) && mod(numel(w),2) == 0
            w = reshape(w, 2, [])';
        % Case 2: N-by-2 matrix
        elseif ismatrix(w) && size(w,2) == 2
            % already fine
        else
            return; % invalid shape
        end
        
        onsets = w(:,1);
        offsets = w(:,2);
        
        % Check onset < offset
        if any(offsets <= onsets)
            return;
        end
        
        % Check for overlap
        [~, order] = sort(onsets);
        onsets = onsets(order);
        offsets = offsets(order);
        
        if any(onsets(2:end) < offsets(1:end-1))
            return;
        end
        
        ok = true;
    end
end

