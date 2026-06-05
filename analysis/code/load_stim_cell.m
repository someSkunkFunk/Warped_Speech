function stim=load_stim_cell(envelopes_flpth,experiment_cond,trial_nums)
% stim=load_stim_cell(envelopes_flpth,experiment_cond,trial_nums)
% loads stimuli in condition order from experiment

if numel(experiment_cond)~=numel(trial_nums)
    error('number of conditions doesnt match numebr of trials.')
end
load(envelopes_flpth,'env')
fprintf('loading envelopes from %s\n',envelopes_flpth)

stim = env(experiment_cond,trial_nums);
% stim = env(preprocessed_eeg.cond,1:preprocess_config.n_trials);
stim = stim(logical(eye(size(stim))));

end
