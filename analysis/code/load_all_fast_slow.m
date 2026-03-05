% load preprocessed eeg for fast-slow into single var
subjs=[2:7,9:22];
n_trials=75;
EEG=cell(length(subjs),n_trials);
STIM=cell(size(EEG));
COND=nan(length(subjs),n_trials);
script_config=[];

script_config.show_tuning_curves=false;
for ss=1:length(subjs)
    subj=subjs(ss);
    trf_analysis_params;
    fprintf('loading subj %d...\n',subj);
    pp_checkpoint_=load_checkpoint(preprocess_config);
    EEG(ss,pp_checkpoint_.preprocessed_eeg.trials)=pp_checkpoint_.preprocessed_eeg.resp;
    % each subj has a different order so we have to package that too
    COND(ss,pp_checkpoint_.preprocessed_eeg.trials)=pp_checkpoint_.preprocessed_eeg.cond;
    STIM(ss,pp_checkpoint_.preprocessed_eeg.trials)=load_stim_cell(trf_config.paths.envelopesFile,COND(ss,pp_checkpoint_.preprocessed_eeg.trials),pp_checkpoint_.preprocessed_eeg.trials);
    
    clear pp_checkpoint_
end
disp('done.')