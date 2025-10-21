function update_rehash_config(new_config)
%assumes foldertree structure mapping to config.paths.output_dir hasn't
%been changed

%read existing registry
registry_file=fullfile(new_config.paths.output_dir,'registry.json');
if ~isfile(registry_file)
    error('registry not found in %s',registry_file)
end

registry=jsondecode(fileread(registry_file));


%loop through each registry/file

%check that existing fields all match between the saved config and the
%intended new config, except for fields that are maybe new

% if so, update registry with new hash

% update existing configs contained in file to include new fields ONLY

% update saved file with new config and rewrite hash in folder name
end