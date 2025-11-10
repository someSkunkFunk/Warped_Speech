%% clean registries
% function clean_registries(subjs)
% TODO: add a bit where it deletes registry entries/hashes without an
% existing mat file
for subj=[2:7,9:22]
    fprintf('cleaning registry for subj %d...\n',subj)
    trf_analysis_params
    % map each registry to a unique matfpth
    % note: assuming each redundant registry only maps to a single file -
    % not sure this is the case

    % identify redundant registry hashes - keep only one and update its
    % contents to reflect the single unique matfpth
    pp_registry_file=fullfile(preprocess_config.paths.output_dir,'registry.json');
    trf_registry_file=fullfile(trf_config.paths.output_dir,'registry.json');
    disp('cleaning preprocessing registry...')
    pp_registry=jsondecode(fileread(pp_registry_file));
    pp_repeats=find_repeat_hashes(pp_registry);
    if isempty(pp_repeats)
        disp('no repeated hashes found')
    else
        pruned_pp_registry=prune_repeat_hashes(pp_registry,pp_repeats);
        %TODO: validate repeats are identical, delete registry entries with
        % old timestamps, save pruned registry to regitry file.
        % save updated registry - note we copied this from save checkpoint
        % which probably means we ought to instead make it it's own
        % independent function in case this process changes
        fid=fopen(trf_registry_file,'w');
        fwrite(fid,jsonencode(pruned_pp_registry),'char');
        fclose(fid);
        fprintf('Updated registry at %s.\n',pp_registry_file);
    end

    disp('cleaning trf registry...')
    trf_registry=jsondecode(fileread(trf_registry_file));
    trf_repeats=find_repeat_hashes(trf_registry);
    if isempty(trf_repeats)
        disp('no repeated hashes found.')
    else
        pruned_trf_registry=prune_repeat_hashes(trf_registry,trf_repeats);
        % save updated registry - note we copied this from save checkpoint
        % which probably means we ought to instead make it it's own
        % independent function in case this process changes
        fid=fopen(trf_registry_file,'w');
        fwrite(fid,jsonencode(pruned_trf_registry),'char');
        fclose(fid);
        fprintf('Updated registry at %s.\n',trf_registry_file);
        % error('need to save updated result still')
    end
    



end


function pruned_registry=prune_repeat_hashes(registry,repeats)
% ensures repeated hashes map to same config/matfile and only differ in 
% timestamp - then removes all but the entry with latest timestamp from
% registry
registry_fields=fieldnames(registry);
all_hashes=unique({registry.hash});

%TODO: preallocate outputregistry, starting with non-repeat entries
%populated - use number of unique hashes as length of output registry
pruned_registry=cell2struct(cell([length(registry_fields), ...
    length(all_hashes)]),registry_fields,1);
non_repeats=setdiff(all_hashes,repeats);
    for hh=1:length(all_hashes)
        hash=all_hashes(hh);
        hash_match_idx=find(strcmp(hash,{registry.hash}));
        if ismember(hash,non_repeats)
            if length(hash_match_idx)>1
                error('%s hash categorized into non-repeats but multiple entries match.',hash)
            else
                pruned_registry(hh)=registry(hash_match_idx);
            end
        elseif ismember(hash,repeats)
            if length(hash_match_idx)==1
                error('%s hash categorized into repeats but only a single entry matches.',hash)
            else
                fprintf(['making sure that n=%d entries with\n' ...
                    '%s hash are fully equivalent.\n'],length(hash_match_idx),hash{:})
                subregistry=rmfield(registry(hash_match_idx),'timestamp');
                match_m=false(size(subregistry));
                match_m(1)=true;
                % note: there must be a way to avoid a loop here...
                for ss=2:length(subregistry)
                    match_m(ss)=isequal(subregistry(ss),subregistry(1));
                end
                if all(match_m)
                    rep_timestamps=cellfun(@(x) datetime(x),{registry(hash_match_idx).timestamp});
                    latest_rep_timestamp=max(rep_timestamps);
                    % if they're all equivalent, can just assing the first
                    % one and the overwrite it's timestamp
                    entry=subregistry(1);
                    entry.timestamp=latest_rep_timestamp;
                    pruned_registry(hh)=entry;
                else
                    error('didnt expect this case and thus unaccounted for...')
                end
            end
        else
            error('idk.')
        end
       
    end
end
function repeated_hashes=find_repeat_hashes(registry)
% finds hash values that occur repeatdly in registry
    [hashes, ~, hash2reg_idx]=unique({registry.hash});
    % count number of times each hash appears in registry - right bin edge
    % is inclusive so we need to rig histogram bin edges
    hash_counts=histcounts(hash2reg_idx,[unique(hash2reg_idx);max(hash2reg_idx+1)]);
    repeated_hashes=hashes(hash_counts>1);
end
