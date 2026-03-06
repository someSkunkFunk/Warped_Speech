% load preprocessed eeg and stim features
% for fast-slow into single var

subjs=[2:7,9:22];
n_trials=75;
EEG=cell(length(subjs),n_trials);
ENV=cell(size(EEG));
PKRT=cell(size(EEG));
COND=nan(length(subjs),n_trials);
script_config=[];
script_config.show_tuning_curves=false;
pkrt=load_pkrt(n_trials);
%%

for ss=1:length(subjs)
    subj=subjs(ss);
    trf_analysis_params;
    fprintf('loading subj %d eeg...\n',subj);
    pp_checkpoint_=load_checkpoint(preprocess_config);
    trials_recorded_=pp_checkpoint_.preprocessed_eeg.trials;
    EEG(ss,trials_recorded_)=pp_checkpoint_.preprocessed_eeg.resp;
    % each subj has a different order so we have to package that too
    COND(ss,trials_recorded_)=pp_checkpoint_.preprocessed_eeg.cond;
    ENV(ss,trials_recorded_)=load_stim_cell(trf_config.paths.envelopesFile, ...
        COND(ss,trials_recorded_),trials_recorded_);
    pkrt_tmp_=pkrt(COND(ss,trials_recorded_),trials_recorded_);
    PKRT(ss,trials_recorded_)=pkrt_tmp_(logical(eye(length(trials_recorded_))));
    
    clear pp_checkpoint_ trials_recorded_ pkrt_tmp_
end
disp('done.')
clear do_nulltest preprocess_config ss subj train_params trf_config
clear user_profile warped_speech_dir pkrt

function pkrt=load_pkrt(n_trials)
global boxdir_mine boxdir_lab 
% helper to hide loading bullshit -- if we can be sure that structure array
% fields preserve order they are stored in, we could just get the rate
% names from fieldname but i'm not certain
rate_names={'fast','og','slow'};
%load peakrate for all trials -- organize by subject in loop
pkrt_path=fullfile(boxdir_mine,'stimuli','wrinkle','peakRate', ...
    'fastSlow.mat');
fprintf('loading peakrate from %s...\n',pkrt_path)
load(pkrt_path) % loads "config_peakrate" (1x1) struct with params, result.(cond) ...
% restructure pkrt into the format that makes sense for indexing
pkrt=cell([length(rate_names),n_trials]);
for nn=1:n_trials
    for cc=1:length(rate_names)
        pkrt{cc,nn}=config_peakrate.result.(rate_names{cc}){nn};
    end
end
end