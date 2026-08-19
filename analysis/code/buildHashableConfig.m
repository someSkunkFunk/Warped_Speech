function hc=buildHashableConfig(config)
% single source of truth for identity-hash computation. Used by
% save_checkpoint.m, load_checkpoint.m, and update_rehash_config.m so there
% is exactly one implementation to keep in sync (replacing olf
% remove_nested_paths / remove_nested_paths_recursive pair)
% 
% only fields listed in getHashFields(config.configType) are hashed.
% Everything else (paths, configType, any analysis-specific bookkeeping
% fields) is excluded by construction. Nested sub-configs that themselves
% declary configType (eg. trf_config.preprocess_config) are filtered
% recursively through their own whitelist too, so a new field added deep
% inside preprocess_config can't silently change a trf-level hash either
    if ~isfield(config,'configType')
        error('computeConfigHash:missing_type', ...
            ['config must have a configType field, set by its builder' ...
            'function'])
    end
    hashFields=getHashFields(config.configType);
    hc=filterConfig(config,hashFields);

    if isfield(hc,'preprocess_config') && ...
        isstruct(hc.preprocess_config) && ...
        isfield(hc.preprocess_config,'configType')
        
        hc.preprocess_config=buildHashableConfig(hc.preprocess_config);
    end
    hc=columnize_row_vectors(orderfields(hc));
end