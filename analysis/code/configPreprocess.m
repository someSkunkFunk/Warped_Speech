function preprocessConfig=configPreprocess(preprocessConfig)
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
    'n_trials',[], ...
    'experiment',[], ...
    'use_triggers','click',...
    'm',[] ...
    );
fields=fieldnames(defaults);
for ff=1:numel(fields)
    if ~isfield(preprocessConfig,fields{ff})||isempty(preprocessConfig.(fields{ff}))
        preprocessConfig.(fields{ff})=defaults.(fields{ff});
    end
end
if preprocessConfig.subj>22
    preprocessConfig.experiment='regIrreg';
elseif preprocessConfig.subj>0&&preprocessConfig.subj<=22
    preprocessConfig.experiment='fastSlow';
else
    error('undefined experiment?')
end

if (preprocessConfig.subj<7||preprocessConfig.subj==22)&& ...
    strcmp(preprocessConfig.use_triggers,'click')
    warning(['click triggers selected for subj %d but subjs 6 and below ' ...
        'dont have click triggers, thus force-changing trigger mode to ' ...
        'psychportaudio.'],preprocessConfig.subj)
    preprocessConfig.use_triggers='psychportaudio';
    
end

%%%%NOTE: WHEN USING CLICK TRIGGERS, WE CAN IGNORE THE SPEECH DELAY ADDED
%%%%BY NOISYSPEECH FUNCTION 
% want to avoid adding stim-delay time
% for subjects without noise- that gets fi
switch preprocessConfig.use_triggers
    case 'click'
        % no speech delay!
        preprocessConfig.stim_delay_time=[];
    case 'psychportaudio'
        % note that some subjects dont have clicks and some have both click
        % and psychport triggers but no delay so this is actually not the
        % most helpful
    otherwise
        error('what trigger did you mean by %s?',preprocessConfig.use_triggers)

end

switch preprocessConfig.ref
    case 'avg'
        preprocessConfig.refI=1:preprocessConfig.nchan;
    case 'mast'
        preprocessConfig.refI=preprocessConfig.nchan+(1:2);
    otherwise
        % specificy which channels in ref
        preprocessConfig.refI=ref;
end
preprocessConfig.opts = {'channels',1:(preprocessConfig.nchan+2),'importannot','off','ref',preprocessConfig.refI};



%%%%%%%%%%%%%%%%%%%%%% PATHS TO IGNORE IN REGISTRY %%%%%%%%%%%%%%%%%%%%%%%%

% package box foldertree stuff into single field we can later ignore in
% datahash:

preprocessConfig.paths.datafolder = sprintf('%s/data/%s', ...
    boxdir_mine,preprocessConfig.experiment);
if preprocessConfig.subj>=90


    preprocessConfig.paths.output_dir=sprintf('%spreprocessed_eeg/%s/s%0.2d', ...
        preprocessConfig.paths.datafolder,preprocessConfig.subj);
    
    
    preprocessConfig.paths.behfile = sprintf('%ss%0.2d_RegIrregPilot.mat', ...
        preprocessConfig.paths.datafolder,preprocessConfig.subj);
    preprocessConfig.paths.bdffile = sprintf('%sbdf/reg_irreg_pilot_s%0.2d.bdf', ...
        preprocessConfig.paths.datafolder,preprocessConfig.subj);

else
    
    preprocessConfig.paths.output_dir=sprintf('%spreprocessed_eeg/s%0.2d', ...
        preprocessConfig.paths.datafolder,preprocessConfig.subj);
    
    
    preprocessConfig.paths.behfile = sprintf('%ss%0.2d_WarpedSpeech.mat', ...
        preprocessConfig.paths.datafolder,preprocessConfig.subj);
    preprocessConfig.paths.bdffile = sprintf('%sbdf/warpedSpeech_s%0.2d.bdf', ...
        preprocessConfig.paths.datafolder,preprocessConfig.subj);
end

preprocessConfig.paths.chanlocs_path=sprintf('%s128chanlocs.mat', ...
    preprocessConfig.paths.datafolder);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% SEE HOW MANY TRIALS RECORDED IN BEHAVIOR FILE %%%%%%%%%%%%%%%
load(preprocessConfig.paths.behfile,'m');
preprocessConfig.m=m;
preprocessConfig.n_trials=length(m);

%% put everything into a structure and disp
preprocessConfig.configType='preprocess';
% because order matters for jsonencode-based match lookup...
preprocessConfig=orderfields(preprocessConfig);
preprocessConfig=columnize_row_vectors(preprocessConfig);
disp('voila preprocess_config:')
disp(preprocessConfig)
end