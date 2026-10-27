function migrateHashSchema(rootDir)
 % miggrateHashSchema: Rehashes and renames checkpoint files after a schema change.
%
%   migrateHashSchema(rootDir) searches for all registry.json files within 
%   ROOT_DIR and its subdirectories. For each registry, it recalculates the 
%   config hash using the updated buildHashableConfig rules (which now exclude 
%   non-parameter metadata like 'experiment'). It then renames the associated 
%   .mat files to match the new hashes and updates the registry.json entries.
%
%   Inputs:
%       rootDir - (string|char) Path to the root output directory containing 
%                  the subject folders and registry.json files.
%
%   Outputs:
%       None. Modifies .mat filenames and registry.json files in place.
%
%   Example:
%       migrate_hash_schema('/path/to/project/output');
%
%   Dependencies:
%       Requires buildHashableConfig and DataHash to be on the MATLAB path.
%
%   See also: buildHashableConfig, progressSave, progressLoad  


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

    if numel(reg) == 1 && ~iscell(reg)
        reg=reshape(reg,1,1); % ensure struct array semantics
    end

    updated=false;
    for jj=1:numel(reg)
        oldHash=reg(jj).hash;
        oldConfig=reg(jj).config;

        % inject configType if missing from config struct
        % since buildHashableConfig requires it (and its probably missing
        % from oldConfig since buildHashableConfig removes it before
        % recording to registry)
        if ~isfield(oldConfig, 'configType') && isfield(reg)
            oldConfig.configType=reg(jj).configType;
        end

        % compute new hashable config using the UPDATED getHashFields
        % this should automatically strip out 'experiment' and other
        % removed fields
        newHc=buildHashableConfig(oldConfig);
        % compute new hash
        newHash=char(upper(DataHash(newHc)));

        if ~strcmp(oldHash,newHash)
            % 1. rename .mat file
            oldMatFile=fullfile(regs(ii).folder, reg(jj).file);
            [~, name, ext]=fileparts(oldMatFile);

            % replace old hash with new hash in filename
            newName=strrep(name,old_hash,new_hash);
            newMatFile=fullfile(regs(ii).folder, [newName, ext]);

            if isfile(oldMatFile)
                movefile(oldMatFile, newMatFile);
                fprintf('Renamed: %s -> %s\n', reg(jj))
            else
                warning('File not found: %s', oldMatFile)
            end

            % 2. update registry entry
            reg(jj).hash=newHash;
            reg(jj).file=[newName, ext];
            reg(jj).config=newHc;

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