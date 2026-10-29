function migrateHashSchema(rootDir,forceOverwrite)
 % miggrateHashSchema: Rehashes and renames checkpoint files after a schema change.
%
%   migrateHashSchema(rootDir) searches for all registry.json files within 
%   rootDIR and its subdirectories. For each registry, it recalculates the 
%   config hash using the updated buildHashableConfig rules (which now exclude 
%   non-parameter metadata like 'experiment'). It then renames the associated 
%   .mat files to match the new hashes and updates the registry.json entries.
%
%   Inputs:
%       rootDir - (string|char) Path to the root output directory containing 
%                  the subject folders and registry.json files.
%       forceOverwrite - bool (false default) force overwriting the mat &
%       registry files
%
%   Outputs:
%       None. Modifies .mat filenames and registry.json files in place.
%
%   Example:
%       migrateHashSchema('/path/to/project/output');
%
%   Dependencies:
%       Requires buildHashableConfig and DataHash to be on the MATLAB path.
%
%   See also: buildHashableConfig, progressSave, progressLoad  

if nargin<2
    forceOverwrite=false;
end

% find all registry.json files in the output directory tree
regs=dir(fullfile(rootDir, '**', 'registry.json'));

if isempty(regs)
    fprintf('No registry.jsonfiles found in %s\n', rootDir)
    return
end

for ii=1:numel(regs)
    regPath=fullfile(regs(ii).folder, regs(ii).name);
    fprintf('Processing %s...\n', regPath)

    regText=fileread(regPath);
    if isempty(strtrim(regText)), continue, end
    reg=jsondecode(regText);

    if isempty(reg), continue, end
    fprintf('%d entries found in registry at %s.\n',numel(reg),regPath)
    if numel(reg) == 1 && ~iscell(reg)
        reg=reshape(reg,1,1); % ensure struct array semantics
    end

    updated=false;
    for jj=1:numel(reg)
        oldHash=reg(jj).hash;
        oldConfig=reg(jj).config;
        oldConfigType=reg(jj).configType;


        % inject configType if missing from config struct
        % since buildHashableConfig requires it (and its probably missing
        % from oldConfig since buildHashableConfig removes it before
        % recording to registry)

        if ~isfield(oldConfig, 'configType') && isfield(reg(jj), 'configType')
            oldConfig.configType=oldConfigType;
        end

        % compute new hashable config (& other registry entry data)
        % using the UPDATED getHashFields
        % this should automatically strip out 'experiment' and other
        % removed fields
        newHc=buildHashableConfig(oldConfig);
        % compute new hash
        newHash=char(upper(DataHash(newHc)));

        % 0: book-keeping
        % note to self: i left the old vars here in case they help
        % debugging after registry has been changed in step 2
        oldHashFields=reg(jj).hashFields;
        oldTimeStamp=reg(jj).timestamp;
        % this should NOT change (as of 20261028)
        newConfigType=oldConfigType;
        newHashFields=getHashFields(newConfigType);

        

        if ~strcmp(oldHash,newHash) || ...
            ~isempty(setdiff(oldHashFields,newHashFields)) || ...
            ~strcmp(newConfigType,oldConfigType) || ...
            forceOverwrite
            % 1. rename .mat file
            oldMatFile=fullfile(regs(ii).folder, reg(jj).file);
            [~, name, ext]=fileparts(oldMatFile);

            % in case extension is missing for some reason
            if ~strcmp(ext,"")&&~strcmp(ext,'.mat')
                error('unknown file type: %s',ext)
            end
            if strcmp(ext,'')
                ext='.mat';
                oldMatFile=sprintf('%s%s',oldMatFile, ext);
            end
            % replace old hash with new hash in filename
            newName=strrep(name,oldHash,newHash);
            newMatFile=fullfile(regs(ii).folder, sprintf('%s%s',newName, ext));

            heedWarning=true;
            if strcmp(oldMatFile,newMatFile)
                heedWarning=false;
            end

            if isfile(oldMatFile)
                movefile(oldMatFile, newMatFile);
                fprintf('Renamed: %s -> %s\n', reg(jj).file, newMatFile);
            else
                if heedWarning
                    warning('File not found: %s', oldMatFile);
                else
                    fprintf('Mat file name is already correct: %s -> %s\n', ...
                        reg(jj).file, sprintf('%s%s',newName, ext));
                    % NOTE: not renaming .mat file because 
                    % isfile(oldMatFile) should only be false when no mat
                    % files in folder have the old hash -- if the opposite
                    % is true and the old hash is in the folder, first
                    % check that oldMatFile is specified correctly before
                    % potentially wrongfully equating separate hashes.
                    warning('Not renaming mat file on purpose, see comment^.')
                end
            end

            % 2. update registry entry
            reg(jj).hash=newHash;
            reg(jj).file=sprintf('%s%s',newName, ext);
            reg(jj).config=newHc;
            reg(jj).hashFields=getHashFields(newConfigType);
            reg(jj).timestamp=datetime('now');


            updated=true;
        end
    end
    if updated
        % save updated registry
        fid=fopen(regPath,'w');
        fwrite(fid,jsonencode(reg),'char');
        fclose(fid);
        disp('Updated registry saved.')
    else
        disp('no changes needed.')
    end
end
disp('Migration complete!')
end
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh
%asdfghjkl;lkjhgfdsasdfghjklkjhgfdsdfghjklkjhgfdsdfghjklkjhgfdsasdfghjklkjh