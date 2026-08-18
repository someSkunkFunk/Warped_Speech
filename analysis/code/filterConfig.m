function hashableConfig=filterConfig(config,keepFields)
    % return copy of config struct with only fields that should affect hash
    % value (analysis parameters)
    hashableConfig=cell2struct(cell(size(keepFields))',keepFields);
    for ff=1:numel(keepFields)
        ffField=keepFields{ff};
        hashableConfig.(ffField)=config.(ffField);
    end
end