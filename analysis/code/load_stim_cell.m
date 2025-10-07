function stim=load_stim_cell(envelopes_flpth,experiment_cond,trial_nums)
% stim=load_stim_cell(envelopes_flpth,experiment_cond,trial_nums)
% loads stimuli in condition order from experiment
%NOTE: COND and preprocessed_eeg.trials will be needed here and not in
%preprocess_config
if numel(experiment_cond)~=numel(trial_nums)
    error('number of conditions doesnt match numebr of trials.')
end
load(envelopes_flpth,'env')
fprintf('loading envelopes from %s\n',envelopes_flpth)
% fs_stim=load(preprocess_config.envelopesFile,'fs');
% fs_stim=fs_stim.fs;
% check wav fs matches analysis fs
% if fs_stim ~= preprocess_config.fs
%     error('stim has wrong fs.')
% end
%TODO: what happens when trial data missing in cond or bdf itself?
%note: we could easily make trials an input variable to flexibly handle
%cases where only subset of trials is desired
stim = env(experiment_cond,trial_nums);
% stim = env(preprocessed_eeg.cond,1:preprocess_config.n_trials);
stim = stim(logical(eye(size(stim))));
% TODO: figure out if this line below was supposed to go in
% clean_false_starts as we presumed?
% RE above: not sure if having this code here is a problem when that
% happens but it's definitely causing problems when there are missing
% trials 
% cond = cond(preprocessed_eeg.trials,1);
% stim = stim(preprocessed_eeg.trials,1)';
end
