function update_rehash_config(new_config)
% UPDATE_REHASH_CONFIG(new_config)
%
% migrates every entry in config.paths.output_dir/registry.json to the
% current hash schema (using getHashFields) and current config-builder
% defaults (config_preprocess/config_trf)
%
% CHANGED FROM OLD VERSION: instead of trying to match one
% externally-supplied config against ssaved files and work with the first
% hit, it rebuilds every registry entry's own stored config by re-running
% it through its original builder function. The builder function fills any
% newly-added fields with schema defaults while leaving pre-existing values
% untouched (using config functions' "if ~isfield||isempty" backfill
% loops). Every saved variant in the directory gets migrated in one call,
% to avoid looping over multiple settings just to construct a new_config
% that happens to match each one.
%
% USAGE:
% trf_analysis_params;
% update_rehash_config(trf_config)
% update_rehash_config(preprocess_config)
%
% config only needs valid .paths.output_dir and .configType. It locates the
% registry, it is NOT ground truth to match against
%
% SAFETY: rebuild is only expcted to ADD fields. If rebuilding an entry
% would change the value of a field that already existed before migration,
% entry is skipped with a warning instead of silently overwritten.

outputDir=config.paths.output_dir;
registryFile=fullfile(output_dir,'registry.json');
if ~isfile(registryFile)
    error('registry not found in %s', registryFile)
end
registry=jsondecode(fileread(registryFile));

% this deals with the scalar-struct case addressed in save/load checkpoint
% when checking elseif numel(registry) == 1 && ~iscell(registry)
% probably unnecessary, but scared to commit to removing it from save/load
% checkpoint without first implementing the first re-hash and addressing
% any bugs that may surface.
% if isstruct(registry)
%     registry=reshape(registry,[],1); %
% end

nUpdated=0; nUnchanged=0; nSkipped=0;
for ii=1:numel(registry)
    entry=registry(ii);
end

end