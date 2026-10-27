function progressSave(data,inputConfig,overwrite)
% PROGRESSSAVE single variable of data saved to a mat file and json-encoded
% registry using hash system to keep track of multiple data variables
% originating from the same analysis parameters in a single mat file
    arguments
        data
        inputConfig (1,1) struct
        overwrite (1,1) logical=false;
    end
    % record var name in outer scope so we can reference it when loading
    varname=inputname(1);
    data_.(varname)=data;
    subj=inputConfig.subj;
    outputDir=inputConfig.paths.output_dir;
    % ENSURE OUTPUT DIRECTORY EXISTS
    if ~exist(outputDir,'dir')
        mkdir(outputDir);
    end
    hashableConfig=buildHashableConfig(inputConfig);
    configHash=computeConfigHash(inputConfig);

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
        newCheckpointVar=~ismember(varname,{whos('-file',matFpth).name});
    else
        newCheckpointVar=false;
    end
    if isempty(config_match_idx)||newCheckpointVar||overwrite
        
        if overwrite
            % desired behavior when overwrite TRUE:
            % if matching config exists, replace its registry entry and mat
            % file with new data
            warning('overwrite set to true... ensure code will do what is intended...')
        end
        disp('original registry:')
        disp(registry)
        fprintf('saving %s\nto %s\nfor subj %02d\n(config hash:%s)\n', ...
            varname,matFpth,subj,configHash)
        % include extension in registry filename entry
        [~,matFnm,matExt]=fileparts(matFpth);
        
        %add or update entry
        entry=struct( ...
            'hash',configHash,...
            'config',hashableConfig, ...
            'hashFields',{getHashFields(inputConfig.configType)}, ...
            'configType',inputConfig.configType, ...
            'file', [matFnm, matExt], ...
            'timestamp',datetime('now'));

        idx = config_match_idx;
        if isempty(idx)
            assert(~isfile(matFpth),['This function assumes the mat' ...
                ' file does not exist without a registry, but ' ...
                'matfile %s exists.'],matFpth)
            % existing entries dont match, add new registry entry at end
            disp('adding new registry entry')
            idx = numel(registry) + 1;
        else
            if overwrite || newCheckpointVar
                disp('rewriting existing registry entry:')
                disp(registry(idx))
            else
                warning(['Existing registry matches input, ' ...
                    'does not appear to be a new variable and ' ...
                    'overwrite is false. Aborting.'])
                return;
            end
        end

        % ADDRESS POSSIBLY DISSIMILAR STRUCTURES FROM SCHEMA CHANGES
        % if new entry has new fields that current registry lacks, add
        % fields to existing registry with empty entried
        if ~isempty(registry)
            regFields=fieldnames(registry);
            entFields=fieldnames(entry);
            newFields=setdiff(entFields,regFields);

            if ~isempty(newFields)
                for ii=1:numel(registry)
                    for jj=numel(newFields)
                        registry(ii).(newFields{jj})=[];
                    end
                end
            end
        end
        
        registry(idx) = entry;

        
        disp('new registry:')
        disp(registry)
        % save data
        if isfile(matFpth)
            % new results with same config to add to existing file
            save(matFpth,'-struct','data_','-append');
        else
            % make new file with results
            save(matFpth,'-struct','data_');
        end
        % overwrite pre-existing config (they shld match -- and hopefully 
        % latest version captures schema at checkpoint saving time)
        config=inputConfig;
        save(matFpth,'config','-append');
        % save updated registry
        fid=fopen(registryFile,'w');
        fwrite(fid,jsonencode(registry),'char');
        fclose(fid);
        fprintf('Saved %s to %s and updated registry.\n',varname,matFpth);
    else
        warning('pre-existing matching config exists, skipping save - ensure that this is intended behavior.')

    end
end
