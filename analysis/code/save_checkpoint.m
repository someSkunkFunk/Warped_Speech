function save_checkpoint(data,config,overwrite)
    arguments
        data
        config (1,1) struct
        overwrite (1,1) logical=false;
    end
    % warning('I think overwrite param is causing repeated entries to registry being recorded... we should fix that...')
    % record var name in outer scope so we can reference it when loading
    varname=inputname(1);
    data_.(varname)=data;
    subj=config.subj;
    output_dir=config.paths.output_dir;
    if ~exist(output_dir,'dir')
        mkdir(output_dir);
    end
    % normalize struct structure so hashes are consistent
    config=remove_nested_paths(config);
    config=columnize_row_vectors(config);
    % make unique hash using DataHash from fileexchange
    config_hash=char(upper(DataHash(config)));

    % define unique file names
    mat_fpth=fullfile(output_dir,sprintf('warped_speech_s%02d_%s.mat',config.subj,config_hash));
    registry_file=fullfile(output_dir,'registry.json');

    % load or initialize registry
    if isfile(registry_file)
        registry=jsondecode(fileread(registry_file));
        config_match_idx=find(strcmp({registry.hash},config_hash),1);
    else
        % intialize as 0x0 struct array for iterative assignment without
        % error
        registry=struct('hash',{}, ...
            'config',{}, ...
            'file',{}, ...
            'timestamp',{} ...
            );
        config_match_idx=[];
    end
    
    %if no matching registry exits, or file associated with registry 
    % is missing current variable, save and register
    if isfile(mat_fpth)
        missing_var=~ismember(varname,{whos('-file',mat_fpth).name});
    else
        missing_var=false;
    end
    if isempty(config_match_idx)||missing_var||overwrite
        if overwrite
            warning('overwrite set to true... ensure code will do what is intended...')
        end
        disp('original registry:')
        disp(registry)
        fprintf('saving %s\nto %s\nfor subj %02d\n(config hash:%s)\n', ...
            varname,mat_fpth,subj,config_hash)
        %add or update entry
        entry=struct( ...
            'hash',config_hash,...
            'config',config, ...
            'file', mat_fpth, ...
            'timestamp',datetime('now'));
        if overwrite||missing_var
            registry(config_match_idx)=entry;
        else
            registry(end+1)=entry;
        end

        
        disp('new registry:')
        disp(registry)
        % save data
        if isfile(mat_fpth)
            save(mat_fpth,'-struct','data_','-append');
        else
            save(mat_fpth,'-struct','data_');
        end
        % should overwrite pre-existing config but that's okay cuz they
        % shld match
        save(mat_fpth,'config','-append');
        % save updated registry
        fid=fopen(registry_file,'w');
        fwrite(fid,jsonencode(registry),'char');
        fclose(fid);
        fprintf('Saved %s to %s and updated registry.\n',varname,mat_fpth);
    else
        warning('pre-existing matching config exists, skipping save - ensure that this is intended behavior.')

    end
end
