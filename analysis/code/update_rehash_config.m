function update_rehash_config(new_config)
%assumes foldertree structure mapping to config.paths.output_dir hasn't
%been changed
% only adds new fields - does not handle fields being removed

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
    
    if configs_match
    % if so, update registry with new hash/config info
        new_config=rmfield(new_config,'paths');
        new_hash=char(upper(DataHash(new_config)));
        old_hash=char(upper(DataHash(old_config)));
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
end
% if none match - notify
warning('no matching configs were found and none were updated.')

    function match=configs_match
        %preallocate
        config1_=cell2struct(cell(size(fieldnames_common)),fieldnames_common);
        config2_=cell2struct(cell(size(fieldnames_common)),fieldnames_common);
        % populate common fields
        for ff=1:length(fieldnames_common)
            config1_.(fieldnames_common{ff})=old_config.(fieldnames_common{ff});
            config2_.(fieldnames_common{ff})=new_config.(fieldnames_common{ff});
        end
        %check for equality
        match=isequal(config1_,config2_);
    end
end