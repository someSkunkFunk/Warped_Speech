function preprocess_config=config_preprocess(subj)
global user_profile
global boxdir_lab
global boxdir_mine
%preprocess params config file

%TODO: allow function to take variable arguments to seletively change
%specific params if necessary when function is called and not have to edit
%this function

%% stuff that might change depending on context

bpfilter = [1 15];
ref = 'mast';
% fs = 128;
fs=441;

if ismember(subj, [5 8 10 12 13 14 15]) %NOTE: after 12, assuming some bad chans to see what the function detects
    interpBadChans=true;
else
    interpBadChans=false; % only seems necessary for subj 5..?
end
%NOTE: this seems like good oportuninty to use varargin.... TODO 
% for now just do selection automatically by default
bad_chans_manually_selected=false;
bad_chans=[]; %TODO: verify findBadChans returns empty by default if none (seems like it should based on my reading of it)

%% stuff that doesn't generally need to change
%TODO: use relative paths so this is no longer necessary
% NOTE: we ended up needing it because we want to track large files on box
% user_profile=getenv('USERPROFILE');


nchan = 128;
% biosig import options
switch ref
    case 'avg'
        refI=1:nchan;
    case 'mast'
        refI=nchan+(1:2);
    otherwise
        % specificy which channels in ref
        refI=ref;
end
opts = {'channels',1:(nchan+2),'importannot','off','ref',refI};
add_speech_delay_corr=true; % add one sec to event latencies before epoching to account for speech delay in noisy stims
rec_dur = 96; % needed for epoching variable-duration trials
n_trials=75; % ideal number - TODO: replace with true numberbefore 
% saving preprocessed data file

if add_speech_delay_corr
    speech_delay_corr_dir='corrected/';
else
    speech_delay_corr_dir='notcorr/';
end
delay_time=1; %in seconds




datafolder = sprintf('%s/data/',boxdir_mine);
matfolder = sprintf('%smat/%g-%g_%s-ref_%dHz/%s',datafolder,bpfilter(1),bpfilter(2),ref,fs,speech_delay_corr_dir);
matfile = sprintf('%swarpedSpeech_s%0.2d.mat',matfolder,subj);
%above 2 replaced by  2 below
preprocessed_eeg_dir=sprintf('%spreprocessed_eeg/',datafolder);
preprocessed_eeg_path=sprintf('%swarped_speech_s%0.2d.mat',preprocessed_eeg_dir,subj);
if ~exist(preprocessed_eeg_dir,'dir')
    fprintf('%s DNE - making dir...\n',preprocessed_eeg_path)
    mkdir(preprocessed_eeg_dir)
end
%TODO: new preprocessed file should contain the eeg but not the stims
%except for perhaps the index values required to select them from the mat
%file containing all of them and then 
behfile = sprintf('%ss%0.2d_WarpedSpeech.mat',datafolder,subj);
bdffile = sprintf('%sbdf/warpedSpeech_s%0.2d.bdf',datafolder,subj);
envelopesFile=sprintf('%s/stimuli/wrinkle/WrinkleEnvelopes%dhz.mat',boxdir_mine,fs);
% for finding chanlocs file:
% (note - probably should copy this into project for reliability)
%NOTE: this file seems to have coordinates rotated incorrectly - should use
%128chanlocs.mat that is copied into analysis in my box folder for project 
% instead
chanlocs_dir=sprintf('%s/Box/Lalor Lab Box/Code library/EEGpreprocessing/',user_profile);
chanlocs_path=sprintf('%schanlocs.xyz',chanlocs_dir);
        
%% put everything into a structure and disp

vars=whos;
preprocess_config=struct();
for nn=1:numel(vars)
    preprocess_config.(vars(nn).name)=eval(vars(nn).name);
end
fprintf('voila preprocess_config:\n')
disp(preprocess_config)
end