%% clean registries
% function clean_registries(subjs)
subjs=[2];
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
    if isempty(pp_repeats)
        disp('no repeated hashes found')
    else
        pp_registry_pruned=prune_repeat_hashes(pp_registry,pp_repeats);
        %TODO: validate repeats are identical, delete registry entries with
        % old timestamps, save pruned registry to regitry file.
        error('need to save updated result still')
    end

    disp('cleaning trf registry...')
    trf_registry=jsondecode(fileread(trf_registry_file));
    trf_repeats=find_repeat_hashes(trf_registry);
    if isempty(trf_repeats)
        disp('no repeated hashes found.')
    else
        trf_registry_pruned=prune_repeat_hashes(trf_registry,trf_repeats);
        error('need to save updated result still')
    end
    
    dum=[];



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
        match_idx=find(strcmp(hash,{registry.hash}));
        if ismember(hash,non_repeats)
            if length(match_idx)>1
                error('wtf.')
            else
                pruned_registry(hh)=registry(match_idx);
            end
        elseif ismember(hash,repepats)
            if length(match_idx)==1
                error('wtf.')
            else
                fprintf(['making sure that n=%d entries with\n' ...
                    '%s hash are fully equivalent.\n'],length(match_idx),rep_hash{:})
                subregistry=rmfield(registry(match_idx),'timestamp');
                match_m=false(size(subregistry));
                match_m(1)=true;
                % note: there must be a way to avoid a loop here...
                for ss=2:length(subregistry)
                    match_m(ss)=isequal(subregistry(ss),subregistry(1));
                end
                if all(match_m)
                    rep_timestamps=cellfun(@(x) datetime(x),{subregistry.timestamp});
                    latest_rep_timestamp=max(rep_timestamps);
                    pruned_registry(hh)=subregistry(find)
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    %TODO: keep oldest timestamp of this repeated hash.
                else
                    error('didnt expect this case and thus unaccounted for...')
                end
            end
        else
            error('idk.')
        end
       dum=[]; 
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
