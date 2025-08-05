% inspect eeg

subj=2;
preprocess_config=config_preprocess(subj);
%% load bdf
eeglab
EEG = pop_biosig(preprocess_config.bdffile,preprocess_config.opts{:});
[ALLEEG, EEG, CURRENTSET]=eeg_store([],EEG,1);
eeglab redraw
%% plot time-domain

% plot topos

% plot psd