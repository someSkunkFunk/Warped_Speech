function updateRehashConfig(newConfig)
% updateRehashConfig(newConfig)
%
% migrates every entry in config.paths.outputDir/registry.json to the
% current hash schema (using getHashFields) and current config-builder
% defaults (configPreprocess/configTrf)
%
% CHANGED FROM OLD VERSION: instead of trying to match one
% externally-supplied config against ssved files and work with the first
% hit, it rebuilds every registry entry's own stored config by re-running
% it through its original builder function. The builder function fills any
% newly-added fields with schema defaults while leaving pre-existing values
% untouched (using config functions' "if ~isfield||isempty" backfill
% loops). Every saved variant in the directory gets migrated in one call,
% to avoid looping over multiple settings just to construct a new_config
% that happens to match each one.
%
% USAGE:
% setTrfAnalysisParams;
% updateRehashConfig(trf_config)
% updateRehashConfig(preprocess_config)
%
% config only needs valid .paths.outputDir and .configType. It locates the
% registry, it is NOT ground truth to match against
%
% SAFETY: rebuild is only expcted to ADD fields. If rebuilding an entry
% would change the value of a field that already existed before migration,
% entry is skipped with a warning instead of silently overwritten.
%  manual inspection of registry and checkpoint files still required to
% ensure file with older timestamp (and less current results that are
% config-equivalent) doesnt overwrite more recent data (delete older 
% conflicting mats before running script)"

outputDir=newConfig.paths.outputDir;
registryFile=fullfile(outputDir,'registry.json');
if ~isfile(registryFile)
    error('registry not found in %s', registryFile)
end
registry=jsondecode(fileread(registryFile));

% this deals with the scalar-struct case addressed in save/load checkpoint
% when checking elseif numel(registry) == 1 && ~iscell(registry)
% probably unnecessary, but scared to commit to removing it from save/load
% checkpoint without first implementing the first re-hash and addressing
% any bugs that may surface.

% ^ idk wtf this guy's talking about
if isstruct(registry)
    registry=reshape(registry,[],1);
end

nUpdated=0; nUnchanged=0; nSkipped=0;
disp('--Registry on file:')
if isscalar(registry)
    registryTableFmt=repmat(registry,2,1);
else
    registryTableFmt=registry;
end
disp(struct2table(registryTableFmt))

for ii=1:numel(registry)
    
    updateRegistry=true;
    updateMatFile=true;
    entry=registry(ii);


    oldConfig=entry.config;
    oldHash=entry.hash;

    fprintf('\n--- entry %d/%d (file: %s, old hash: %s) ---\n', ...
        ii,numel(registry),entry.file,oldHash);
    fprintf('current outPutDir (%s):\n',outputDir)
    disp(ls(outputDir))

    if ~isfield(oldConfig,'configType')
        % pre-migration entries wont have this field yet... add it based on
        % the type of config passed in to this call
        oldConfig.configType=newConfig.configType;
    end

    % config trf removes subj info when making it a hashable
    % config, so need to temporarily re-instate to avoid error, g et from
    % directory since in the name
    if ~isfield(oldConfig,'subj')
        [~, subjDir_]=fileparts(outputDir);
        oldConfig.subj=double(string(subjDir_));
        clear subjDir_
    end

    try
        newConfig=rebuildConfig(oldConfig);
    catch ME
        warning ('entry %d (%s): could not rebuild config - %s. Skipping.', ...
            ii, entry.file,ME.message)
        nSkipped=nSkipped+1;
        continue
    end
    conflictFields=findConflictingFields(oldConfig,newConfig);

    if ~isempty(conflictFields) && ...
        any(ismember(conflictFields,getHashFields(newConfig.configType)))
        warning(['entry %d (%s): rebuilding changed existing field(s): ' ...
            '{%s}\n check this apparent semantic change. Skipping.'], ...
            ii,entry.file,strjoin(conflictFields,','));
        nSkipped=nSkipped+1;
        continue
    end

    newHash=computeConfigHash(newConfig);
    if strcmp(newHash,oldHash)
        nUnchanged=nUnchanged+1;
        continue
    end

    % rename mat file name in registry and actual file
    % note: dont use the file path in the registry entry because some were
    % recorded wrong
    oldFnm=sprintf('warped_speech_s%02d_%s',oldConfig.subj,oldHash);
    oldMatFpth=fullfile(outputDir,sprintf('%s.mat',oldFnm));
    newFnm=sprintf('warped_speech_s%02d_%s',newConfig.subj,newHash);
    newMatFpth=fullfile(outputDir,sprintf('%s.mat',newFnm));

    if ~isfile(oldMatFpth)
        % rewrite registry entry if matfile was previously updated 
        if isfile(newMatFpth)
            updateRegistry=true;
            updateMatFile=false;
        else
            warning('entry %d: expected file %s not found. Skipping.',ii,oldMatFpth);
            nSkipped=nSkipped+1;
            continue
        end
    end
    
    disp('old config:'); disp(oldConfig);

    disp('new (rebuilt) config:'); disp(newConfig);
    if updateMatFile
        config=newConfig; %#ok<NASGU> % variable named "config" so save() writes it under that name
        save(oldMatFpth,'config','-append')
        
            

        if ~movefile(oldMatFpth,newMatFpth)
            warning(['entry %d: movefile failed (%s -> %s). ' ...
                'Skipping registry update.'],oldMatFpth,newMatFpth);
            nSkipped=nSkipped+1;
            continue
        end
    
    end

    if updateRegistry
        disp('--Registry BEFORE updating:')
        disp(struct2table(registry))
        registry(ii).hash=newHash;
        registry(ii).configType=newConfig.configType;
        registry(ii).hashFields=getHashFields(newConfig.configType);
        registry(ii).config=newConfig;
        registry(ii).file=newFnm;
        registry(ii).timestamp=datetime('now');
        disp('--Registry AFTER updating:')
        disp(struct2table(registry))

    end
    nUpdated=nUpdated+1;
    fprintf('updated: %s -> %s\n',oldHash,newHash);
    
    

end


fid=fopen(registryFile,'w');
fwrite(fid,jsonencode(registry),'char');
fclose(fid);

fprintf(['Done with %s.\n %d updated, %d unchanged, %d skipped ' ...
    '(see warnings aboved for skipped entries)\n'], ...
    outputDir,nUpdated,nUnchanged,nSkipped)

    function newConfig=rebuildConfig(oldConfig)
        switch oldConfig.configType
            case 'preprocess'
                newConfig=configPreprocess(oldConfig);
            case 'trf'
                % re-derive the nested preprocess config first
                ppSeed=oldConfig.preprocess_config;
                ppSeed.configType='preprocess';

                ppSeed.subj=oldConfig.subj;
                ppRebuilt=configPreprocess(ppSeed);
                newConfig=configTrf(oldConfig,ppRebuilt);
            otherwise
                error(['no rebuild rule defined for configType%s\n' ...
                    '--add a case to rebuild_config'],oldConfig.configType);
        end
    end
    function conflictFields=findConflictingFields(oldC,newC)
        conflictFields={};
        % fields to exclude from raw check:
        % 'paths' legitimately differ across machines and is regenerated
        % deterministically; 
        % 'preprocess_config' / 'configType' are structural, not user
        % parameters -- although am suspicious this may trigger a silent
        % bug for trf config results with legitimately different
        % preprocessing params... should verify that's not the case
        exclude={'paths','preprocess_config','configType'};
        checkFields=setdiff(fieldnames(oldC),exclude);
        for ff=1:numel(checkFields)
            fn=checkFields{ff};
            if isfield(newC,fn) && ~isequal(oldC.(fn),newC.(fn))
                conflictFields{end+1}=fn; %#ok<AGROW>
            end
        end
        


    end
end