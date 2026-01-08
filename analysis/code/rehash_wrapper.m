% script to remove paths from existing trf analysis output data so that
% hashes line up across computers - and rewriting said hashes
script_config.show_tuning_curves=false;

% initialize
for subj=[2:7,9:23,96 :98]
% for subj=[9:23,96:98]
trf_analysis_params;
update_rehash_config(trf_config)
end  
%