function preprocess_config=config_preprocess(preprocess_config)
global user_profile
global boxdir_lab
global boxdir_mine
%generate preprocess params config file
defaults=struct( ...
    'subj',[],...
    'bpfilter',[1 15], ...
    'ref','mast', ...
    'fs',128, ...
    'interpBadChans',false, ...
    'bad_chans',[],...
    'manually_selected_bad_chans',false,...
    'nchan',128,...
    'stim_delay_time',1,...
    'epoch_dur',96,...
    'n_trials',75 ...
    );
fields=fieldnames(defaults);
for ff=1:numel(fields)
    if ~isfield(preprocess_config,fields{ff})||isempty(preprocess_config.(fields{ff}))
        preprocess_config.(fields{ff})=defaults.(fields{ff});
    end
end
%% stuff that might change depending on context

% bpfilter = [1 15];
% ref = 'mast';
% fs = 128;
% fs=441;

% if ismember(subj, [5 8 10 12 13 14 15]) %NOTE: after 12, assuming some bad chans to see what the function detects
%     interpBadChans=true;
% else
%     interpBadChans=false; % only seems necessary for subj 5..?
% end
%NOTE: this seems like good oportuninty to use varargin.... TODO 
% for now just do selection automatically by default
% bad_chans_manually_selected=false;
% bad_chans=[]; %TODO: verify findBadChans returns empty by default if none (seems like it should based on my reading of it)

%% stuff that doesn't generally need to change
%TODO: use relative paths so this is no longer necessary
% NOTE: we ended up needing it because we want to track large files on box
% user_profile=getenv('USERPROFILE');


% nchan = 128;
% biosig import options
switch preprocess_config.ref
    case 'avg'
        preprocess_config.refI=1:preprocess_config.nchan;
    case 'mast'
        preprocess_config.refI=preprocess_config.nchan+(1:2);
    otherwise
        % specificy which channels in ref
        preprocess_config.refI=ref;
end
preprocess_config.opts = {'channels',1:(preprocess_config.nchan+2),'importannot','off','ref',preprocess_config.refI};
% add_speech_delay_corr=true; % add one sec to event latencies before epoching to account for speech delay in noisy stims
% rec_dur = 96; % needed for epoching variable-duration trials
% n_trials=75; % ideal number - TODO: replace with true numberbefore 
% saving preprocessed data file

% if add_speech_delay_corr
%     speech_delay_corr_dir='corrected/';
% else
%     speech_delay_corr_dir='notcorr/';
% end
% delay_time=1; %in seconds



% package box foldertree stuff into single field we can later ignore in
% json encode:
preprocess_config.paths.datafolder = sprintf('%s/data/',boxdir_mine);

preprocess_config.paths.output_dir=sprintf('%spreprocessed_eeg/s%0.2d', ...
    preprocess_config.paths.datafolder,preprocess_config.subj);


preprocess_config.paths.behfile = sprintf('%ss%0.2d_WarpedSpeech.mat', ...
    preprocess_config.paths.datafolder,preprocess_config.subj);
preprocess_config.paths.bdffile = sprintf('%sbdf/warpedSpeech_s%0.2d.bdf', ...
    preprocess_config.paths.datafolder,preprocess_config.subj);
preprocess_config.paths.chanlocs_path=sprintf('%s128chanlocs.mat', ...
    preprocess_config.paths.datafolder);
%% put everything into a structure and disp

disp('voila preprocess_config:')
disp(preprocess_config)
end