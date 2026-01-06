% script to remove paths from existing trf analysis output data so that
% hashes line up across computers - and rewriting said hashes


% initialize
for subj=[2]
trf_analysis_params;
update_rehash_config(trf_config)
end
%