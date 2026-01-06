function update_rehash_config(new_config)
% UPDATE_REHASH_CONFIG Update stored config + hash for existing output files.
%
% This function searches the output directory specified by
% new_config.paths.output_dir for saved .mat files containing a `config`
% struct. If a saved config matches the new config on all *existing*
% (common) fields, it:
%   1) Recomputes the config hash using DataHash (excluding `paths`)
%   2) Updates the stored config inside the .mat file
%   3) Renames the file to reflect the new hash
%   4) Updates the corresponding entry in registry.json
%
% Only *new fields* may be added to the config. Field removals or changes
% to existing fields are not supported and will cause no update.
%
% Assumptions:
%   - The folder/file naming scheme has not changed
%   - registry.json exists in output_dir
%   - Each .mat file contains a variable named `config`
%   - Hashes are generated using DataHash on config without `paths`
%
% If no matching config is found, the function exits with a warning.

%read existing registry
registry_file=fullfile(new_config.paths.output_dir,'registry.json');
output_dir=new_config.paths.output_dir;

if ~isfile(registry_file)
    error('registry not found in %s',registry_file)
end

registry=jsondecode(fileread(registry_file));
%note: no need to remove paths since saved configs don't include
fieldnames_new=fieldnames(new_config);
%loop through each file linked to registry
D=dir(sprintf('%s/*.mat',output_dir));
for dd=1:length(D)
    S_=load(fullfile(output_dir,D(dd).name));
    old_config=S_.config;
    clear S_
%check that existing config fields all match between the saved config and the
%intended new config, except for fields that are maybe new
    fieldnames_old=fieldnames(old_config);
    fieldnames_common=intersect(fieldnames_old,fieldnames_new);
    % remove paths fro
    new_config=rmfield(new_config,'paths');
    new_hash=char(upper(DataHash(new_config)));
    old_hash=char(upper(DataHash(old_config)));
    % remove paths from any nested structures in old config NOTE this needs
    % to happen AFTER getting old hash otherwise they might match and
    % nothing gets replaced
    old_config=remove_nested_paths(old_config);

    
    if configs_match(old_config,new_config)
        replace_configs(old_config,new_config)
    else
        % if shared fields dont match - notify
        warning('no matching configs were found and none were updated.')
    end
end

    function old_config=remove_nested_paths(old_config)
        % check if new_config has paths within a nested structure
        for ff=1:length(fieldnames_old)
            field_contents=old_config.(fieldnames_old{ff});
            if isstruct(field_contents) && isfield(field_contents,'paths')
                % overwrite nested structure without paths field
                warning('found paths in nested structure within old_config:')
                disp(field_contents)
                old_config.(fieldnames_old{ff})=rmfield(field_contents,'paths');
            end
        end
    end
    function replace_configs(old_config,new_config)
        % update registry with new hash/config info
        if strcmp(new_hash,old_hash)
            warning('configs must be equal because hashes match. not doing anything.')
            return
        end
        old_hash_idx=find(strcmp({registry.hash},old_hash),1);
        % save results
        config=new_config;
    % update existing config contained in OLD file, then rename file with
    % new hash
        old_mat_fpth=fullfile(output_dir,sprintf('warped_speech_s%02d_%s.mat',config.subj,old_hash));
        new_mat_fpth=fullfile(output_dir,sprintf('warped_speech_s%02d_%s.mat',config.subj,new_hash));
        % update saved file with new config and rewrite hash in folder name
        disp('overwritting old config:')
        disp(old_config)
        disp('with new config:')
        disp(new_config)
        
        save(old_mat_fpth,'config','-append');
        if movefile(old_mat_fpth,new_mat_fpth)
            disp('file rename successful')
            % save updated registry
            disp('old registry:')
            disp(registry)
            disp('new registry:')
            registry(old_hash_idx).hash=new_hash;
            registry(old_hash_idx).config=new_config;
            registry(old_hash_idx).file=new_mat_fpth;
            registry(old_hash_idx).timestamp=datetime('now');
            disp(registry)
            fid=fopen(registry_file,'w');
            fwrite(fid,jsonencode(registry),'char');
            fclose(fid);
            disp('updated registry.');
            return
        else
            error('movefile unsuccessful...')
        end
    end
    function match=configs_match(config1,config2)
        %preallocate
        config1_=cell2struct(cell(size(fieldnames_common)),fieldnames_common);
        config2_=cell2struct(cell(size(fieldnames_common)),fieldnames_common);
        % populate common fields onlty
        for ff=1:length(fieldnames_common)
            config1_.(fieldnames_common{ff})=config1.(fieldnames_common{ff});
            config2_.(fieldnames_common{ff})=config2.(fieldnames_common{ff});
        end
        %check for equality
        match=isequal(config1_,config2_);
    end
end