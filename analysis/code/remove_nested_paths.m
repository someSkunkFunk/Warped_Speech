
%NOTE: this can probably be improved with recursion as it currently only
%removes paths 2 levels down - nesting more than 2 structures might
%preserve paths...
function config=remove_nested_paths(config)
    % base case: no paths field or other substructures
    % struct_mask=structfun(@isstruct,config);
    if ~isfield(config,'paths')&&~any(structfun(@isstruct,config))
        return
    else
        if isfield(config,'paths')
            config=rmfield(config,'paths');
        end
        % need to re-evaluate structure mask after removing paths
        sf_idx=find(structfun(@isstruct,config));
        fldnms=fieldnames(config);
        for sf=1:length(sf_idx)
            ss=sf_idx(sf);
            config.(fldnms{ss})=remove_nested_paths(config.(fldnms{ss}));
        end
    end
end

    
    
    % % check if new_config has paths within a nested structure
    % fldnms=fieldnames(config);
    % for ff=1:length(fldnms)
    %     field_contents=config.(fldnms{ff});
    %     if isstruct(field_contents) && isfield(field_contents,'paths')
    %         % overwrite nested structure without paths field
    %         warning('found paths in nested structure within old_config:')
    %         disp(field_contents)
    %         config.(fldnms{ff})=rmfield(field_contents,'paths');
    %     end
    % end
% end