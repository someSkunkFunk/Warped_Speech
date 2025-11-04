%% clean registries
function clean_registries(subjs)
for subj=subjs
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
    if any(pp_repeats)
        pp_registry_pruned=prune_repeat_hashes(pp_registry,repeat_hashes);
        %TODO: validate repeats are identical, delete registry entries with
        % old timestamps, save pruned registry to regitry file.
    end

    disp('cleaning trf registry...')
    trf_registry=jsondecode(fileread(trf_registry_file));
    trf_repeats=find_repeat_hashes(trf_registry);
    
    dum=[];



end
    function prune_repeat_hashes(registry,repeats)
        disp('making sure they ')
    end
end
function find_repeat_hashes(registry)
    [hashes, ~, hash2reg_idx]=unique({registry.hash});
    % count number of times each hash appears in registry - right bin edge
    % is inclusive so we need to rig histogram bin edges
    hash_counts=histcounts(hash2reg_idx,[unique(hash2reg_idx);max(hash2reg_idx+1)]);
    repeated_hashes=hashes(hash_counts>1);
end
end