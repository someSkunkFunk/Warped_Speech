function stim=load_stim_cell(preprocess_config,preprocessed_eeg)
%NOTE: COND and preprocessed_eeg.trials will be needed here and not in preprocess_config
load(preprocess_config.envelopesFile,'env')
fprintf('loading envelopes from %s\n',preprocess_config.envelopesFile)
fs_stim=load(preprocess_config.envelopesFile,'fs');
fs_stim=fs_stim.fs;
% check wav fs matches analysis fs
if fs_stim ~= preprocess_config.fs
    error('stim has wrong fs.')
end
stim = env(preprocessed_eeg.cond,preprocessed_eeg.trials);
% stim = env(preprocessed_eeg.cond,1:preprocess_config.n_trials);
stim = stim(logical(eye(size(stim))));
% TODO: figure out if this line below was supposed to go in
% clean_false_starts as we presumed?
% RE above: not sure if having this code here is a problem when that
% happens but it's definitely causing problems when there are missing
% trials and we've since altered the code to address that issue by making
% sure cond/trials in preprocessed_eeg only included trials that are not
% missing... theoretically should fix that issue AND false start
% repetitions (since those get cleaned out in preprocessing step too)
% ... watch
% out 
% cond = cond(preprocessed_eeg.trials,1);
% stim = stim(preprocessed_eeg.trials,1)';
end
