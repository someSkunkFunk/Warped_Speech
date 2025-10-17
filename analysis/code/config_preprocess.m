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


% package box foldertree stuff into single field we can later ignore in
% datahash:
if preprocess_config.subj>=90
    preprocess_config.paths.datafolder = sprintf('%s/data/',boxdir_mine);

    preprocess_config.paths.output_dir=sprintf('%spreprocessed_eeg/s%0.2d', ...
        preprocess_config.paths.datafolder,preprocess_config.subj);
    
    
    preprocess_config.paths.behfile = sprintf('%ss%0.2d_RegIrregPilot.mat', ...
        preprocess_config.paths.datafolder,preprocess_config.subj);
    preprocess_config.paths.bdffile = sprintf('%sbdf/reg_irreg_pilot_s%0.2d.bdf', ...
        preprocess_config.paths.datafolder,preprocess_config.subj);

else
    preprocess_config.paths.datafolder = sprintf('%s/data/',boxdir_mine);
    
    preprocess_config.paths.output_dir=sprintf('%spreprocessed_eeg/s%0.2d', ...
        preprocess_config.paths.datafolder,preprocess_config.subj);
    
    
    preprocess_config.paths.behfile = sprintf('%ss%0.2d_WarpedSpeech.mat', ...
        preprocess_config.paths.datafolder,preprocess_config.subj);
    preprocess_config.paths.bdffile = sprintf('%sbdf/warpedSpeech_s%0.2d.bdf', ...
        preprocess_config.paths.datafolder,preprocess_config.subj);
end

preprocess_config.paths.chanlocs_path=sprintf('%s128chanlocs.mat', ...
    preprocess_config.paths.datafolder);
%% put everything into a structure and disp
% because order matters for jsonencode-based match lookup...
preprocess_config=orderfields(preprocess_config);
disp('voila preprocess_config:')
disp(preprocess_config)
end