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
    outputDir=config.paths.output_dir;
    % ENSURE OUTPUT DIRECTORY EXISTS
    if ~exist(outputDir,'dir')
        mkdir(outputDir);
    end
    % normalize struct structure so hashes are consistent
    % config=remove_nested_paths(config);
    % configToHash=filterConfig(config,getHashFields(config.configType));
    % configToHash=columnize_row_vectors(configToHash);
    % config_hash=char(upper(DataHash(configToHash)));
    configHash=computeConfigHash(config);

    % define unique file names
    matFpth=fullfile(outputDir,sprintf('warped_speech_s%02d_%s.mat', ...
        subj,configHash));
    registryFile=fullfile(outputDir,'registry.json');

    % load or initialize registry
    if isfile(registryFile)
        registry=jsondecode(fileread(registryFile));

        % prevent bug if saved registry is empty or has only one element
        if isempty(registry) || ~isstruct(registry)        
            registry=struct('hash',{}, ...
            'config',{}, ...
            'hashFields',{}, ...
            'configType',{}, ...
            'file',{}, ...
            'timestamp',{} ...
            );
        elseif numel(registry) == 1 && ~iscell(registry)
            registry = reshape(registry, 1, 1); % ensure struct array semantics stay consistent
        end
        config_match_idx=find(strcmp({registry.hash},configHash),1);
    else
        % intialize as 0x0 struct array for iterative assignment without
        % error
        registry=struct('hash',{}, ...
            'config',{}, ...
            'hashFields',{}, ...
            'configType',{}, ...
            'file',{}, ...
            'timestamp',{} ...
            );
        config_match_idx=[];
    end
    
    %if no matching registry exits, or file associated with registry 
    % is missing current variable, save and register
    if isfile(matFpth)
        missing_var=~ismember(varname,{whos('-file',matFpth).name});
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
            varname,matFpth,subj,configHash)
        [~,mat_fnm,~]=fileparts(matFpth);
        %add or update entry
        entry=struct( ...
            'hash',configHash,...
            'config',configToHash, ...
            'hashFields',{getHashFields(config.configType)}, ...
            'configType',config.configType, ...
            'file', mat_fnm, ...
            'timestamp',datetime('now'));

        idx = config_match_idx;
        if isempty(idx)
            idx = numel(registry) + 1;
        end
        registry(idx) = entry;

        
        disp('new registry:')
        disp(registry)
        % save data
        if isfile(matFpth)
            save(matFpth,'-struct','data_','-append');
        else
            save(matFpth,'-struct','data_');
        end
        % should overwrite pre-existing config but that's okay cuz they
        % shld match
        save(matFpth,'configToHash','-append');
        % save updated registry
        fid=fopen(registryFile,'w');
        fwrite(fid,jsonencode(registry),'char');
        fclose(fid);
        fprintf('Saved %s to %s and updated registry.\n',varname,matFpth);
    else
        warning('pre-existing matching config exists, skipping save - ensure that this is intended behavior.')

    end
end
