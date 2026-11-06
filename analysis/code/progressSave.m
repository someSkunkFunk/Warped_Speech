function progressSave(data,inputConfig,overwrite)
% PROGRESSSAVE single variable of data saved to a mat file and json-encoded
% registry using hash system to keep track of multiple data variables
% originating from the same analysis parameters in a single mat file.
% overwrite=true replaces that variable in the mat file (other variables
% under the same config are kept) and rewrites the registry entry.
    arguments
        data
        inputConfig (1,1) struct
        overwrite (1,1) logical=false;
    end
    varname=inputname(1);
    if isempty(varname)
        error('progressSave:noVarName', ...
            'First input must be a plain variable name, not an expression.');
    end
    data_.(varname)=data;
    subj=inputConfig.subj;
    outputDir=inputConfig.paths.outputDir;
    if ~exist(outputDir,'dir')
        warning('MAKING NEW OUTPUT DIRECTORY: %s',outputDir)
        mkdir(outputDir);
    end
    hashableConfig=buildHashableConfig(inputConfig);
    configHash=computeConfigHash(inputConfig);

    matFpth=fullfile(outputDir,sprintf('warped_speech_s%02d_%s.mat', ...
        subj,configHash));
    registryFile=fullfile(outputDir,'registry.json');

    % ---- load registry ----
    emptyReg=struct('hash',{},'config',{},'hashFields',{}, ...
        'configType',{},'file',{},'timestamp',{});
    if isfile(registryFile)
        registry=jsondecode(fileread(registryFile));
        if isempty(registry)
            registry=emptyReg;
        elseif iscell(registry)
            % jsondecode returns a cell when entries have different
            % fields; unify rather than discarding the registry
            registry=unifyCellRegistry(registry);
        elseif ~isstruct(registry)
            error('progressSave:badRegistry', ...
                'Unrecognized registry format in %s. Not overwriting.', ...
                registryFile);
        end
        registry=registry(:).'; % consistent row struct array
        config_match_idx=find(strcmp({registry.hash},configHash),1);
    else
        registry=emptyReg;
        config_match_idx=[];
    end

    % ---- determine state ----
    fileExists=isfile(matFpth);
    entryExists=~isempty(config_match_idx);
    if fileExists
        newCheckpointVar=~ismember(varname,{whos('-file',matFpth).name});
    else
        newCheckpointVar=false;
    end

    % skip only if everything is already in place and not overwriting
    if entryExists && fileExists && ~newCheckpointVar && ~overwrite
        warning(['pre-existing matching config exists, skipping save - ' ...
            'ensure that this is intended behavior.'])
        return;
    end

    % ---- warnings for overwrite cases ----
    if overwrite
        if ~fileExists && ~entryExists
            warning(['overwrite=true but no existing mat file or registry ' ...
                'entry found for %s. Added a new entry.'],configHash)
        elseif ~fileExists && entryExists
            warning(['overwrite=true but mat file %s not found (stale ' ...
                'registry entry). Created new file and rewrote entry.'], ...
                matFpth)
        elseif fileExists && ~entryExists
            warning(['overwrite=true and mat file %s exists without a ' ...
                'registry entry (orphaned). Replacing %s and ' ...
                'registering it.'],matFpth,varname)
        else
            warning('overwrite=true: replacing %s in %s and its registry entry.', ...
                varname,matFpth)
        end
    elseif ~entryExists && fileExists
        error('progressSave:orphanFile',['Mat file %s exists without a ' ...
            'registry entry. Use overwrite=true to replace and register it.'], ...
            matFpth)
    end

    fprintf('saving %s\nto %s\nfor subj %02d\n(config hash:%s)\n', ...
        varname,matFpth,subj,configHash)

    % ---- build entry ----
    [~,matFnm,matExt]=fileparts(matFpth);
    entry=struct( ...
        'hash',configHash, ...
        'config',hashableConfig, ...
        'hashFields',{getHashFields(inputConfig.configType)}, ...
        'configType',inputConfig.configType, ...
        'file',[matFnm,matExt], ...
        'timestamp',datetime('now'));

    % ---- align schemas in both directions ----
    if isempty(registry)
        registry=entry;
        idx=1;
    else
        [registry,entry]=alignFields(registry,entry);
        if entryExists
            idx=config_match_idx;
            disp('rewriting existing registry entry:')
            disp(registry(idx))
        else
            idx=numel(registry)+1;
            disp('adding new registry entry')
        end
        registry(idx)=entry;
    end

    % ---- save mat file ----
    if fileExists
        save(matFpth,'-struct','data_','-append');
    else
        save(matFpth,'-struct','data_');
    end
    config=inputConfig;
    save(matFpth,'config','-append');

    % ---- save registry ----
    fid=fopen(registryFile,'w');
    if fid<0
        error('progressSave:registryWrite', ...
            'Could not open %s for writing; mat file saved but registry NOT updated.', ...
            registryFile);
    end
    cleanup=onCleanup(@() fclose(fid));
    fwrite(fid,jsonencode(registry),'char');
    clear cleanup
    fprintf('Saved %s to %s and updated registry.\n',varname,matFpth);
end

function reg=unifyCellRegistry(c)
% convert cell array of structs with differing fields to a struct array
    allFields=cellfun(@fieldnames,c(:),'UniformOutput',false);
    allFields=unique(vertcat(allFields{:}),'stable');
    reg=repmat(cell2struct(cell(numel(allFields),1),allFields,1),1,numel(c));
    for k=1:numel(c)
        f=fieldnames(c{k});
        for m=1:numel(f)
            reg(k).(f{m})=c{k}.(f{m});
        end
    end
end

function [reg,ent]=alignFields(reg,ent)
% add any missing fields (as []) to both registry and entry
    newInReg=setdiff(fieldnames(ent),fieldnames(reg),'stable');
    for jj=1:numel(newInReg)
        [reg.(newInReg{jj})]=deal([]);
    end
    newInEnt=setdiff(fieldnames(reg),fieldnames(ent),'stable');
    for jj=1:numel(newInEnt)
        ent.(newInEnt{jj})=[];
    end
    ent=orderfields(ent,reg);
end