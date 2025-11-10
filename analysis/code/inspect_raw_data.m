% consider filtering triggers so only unpause/pause/click triggers remain
% ^ might make pop_eegplot output more easily readable but not necessary
% really

clear, close all
% todo: add optional lowpass?
subj=22;
preprocess_config.subj=subj;
preprocess_config=config_preprocess(preprocess_config);

inspect_config=[];
inspect_config.show_biosig=true;
inspect_config.highpass=1; % consider going higher...? but also maybe the problem could be low freq stuff....
inspect_config.lowpass=[];
inspect_config.downsample='noAA';
inspect_config.fs=128; % only relevant if downsampling
inspect_config=config_inspect(inspect_config);
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
% load in the data
if inspect_config.skip_rereference
    disp('updating preprocess config to skip reference...')
    % opts is a 'varname', varvalue paired-cell, so have to do this
    % "skipping" indexing
    half_ref_idx_=find(strcmp(preprocess_config.opts(1:2:end),'ref'));
    preprocess_config.opts(half_ref_idx_*2)=[];
    preprocess_config.opts(half_ref_idx_*2-1)=[];
    clear half_ref_idx_
    % this doesn't actually matter but just so reading the config doesnt
    % throw me off...
    preprocess_config.ref='none';
    preprocess_config.refI=[];
    disp(preprocess_config)
end
EEG=pop_biosig(preprocess_config.paths.bdffile,preprocess_config.opts{:});
% plot mastoids sum before filtering - to check if they're average
% referenced or not (will be zero if so)
figure, plot(EEG.data(129,:)+EEG.data(130,:))
title('mastoids sum before filtering + detrending')
%% filter + downsample the data

%NOTE: to replicate pop_downsample vs noAA error, we should keep the order
%of operations used in each case the ssame - namely, pop_downsample was
%used before additional filtering, while noAA was applied after filtering
%(because we were banking on lowpass filt being sufficient antialias)
if ~isempty(inspect_config.downsample)&&strcmp(inspect_config.downsample,'pop')
    disp('downsampling using pop_resample')
    EEG=pop_resample(EEG,inspect_config.fs);
end

if ~isempty(inspect_config.highpass)
    fprintf('hp filtering the shit Fc=%d...\n',inspect_config.highpass)
    hd_hp=getHPFilt(EEG.srate,inspect_config.highpass);
    EEG.data=filtfilthd(hd_hp,EEG.data')';
end
if ~isempty(inspect_config.lowpass)
    fprintf('lp filtering the shit Fc=%d...\n',inspect_config.lowpass)
    hd_lp=getLPFilt(EEG.srate,inspect_config.lowpass);
    EEG.data=filtfilthd(hd_lp,EEG.data')';
end

if ~isempty(inspect_config.downsample)&&strcmp(inspect_config.downsample,'noAA')
    disp('downsampling using noAA.')
    EEG=pop_downsample_noAA(EEG,inspect_config.fs);
end
%% detrend data
if inspect_config.detrend
    disp('detrending data...')
    data_=detrend(EEG.data');
    EEG.data=data_';
    clear data_
end
figure, plot(EEG.data(129,:)+EEG.data(130,:))
title('mastoids sum after filtering')
[ALLEEG, EEG, CURRENTSET]=eeg_store([],EEG,1);
if isnumeric(inspect_config.chns)
    if ~isrow(inspect_config.chns)
        inspect_config.chns=reshape(inspect_config.chns,1,[]);
    end
    plot_chns=sort(inspect_config.chns,2,'descend');
elseif isequal('all',inspect_config.chns)
    plot_chns=[130:-1:1];
end

if inspect_config.show_biosig
    pop_eegplot( EEG, 1, 1, 1);
    eeglab redraw
end
% plot timeseries using regular plot
% start with mastoids

for ee=plot_chns
    % todo: figure -> plot channel with somefixed params, then wait for
    % user input to either move onto next channel OR reframe plot
   ; 
end
% 
% plot spectra
function config=config_inspect(config)
    %NOTES: hard to decode without detrending first
    % could also help to use regular matlab plot but dont do for all chans
    % at once cuz matlab will freak
    defaults=struct('chns', 1:130, ... todo: add option to plot mastoids first
        'show_biosig',false, ...
        'detrend',true, ...
        'highpass',[], ...
        'lowpass',[],...
        'downsample',[],...,
        'fs',[],...
        'skip_rereference',true...
        );
    fields=fieldnames(defaults);
    for ff=1:numel(fields)
        if ~isfield(config,fields{ff})||isempty(config.(fields{ff}))
            config.(fields{ff})=defaults.(fields{ff});
        end
    end
end