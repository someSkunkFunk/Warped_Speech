% consider filtering triggers so only unpause/pause/click triggers remain
subj=98;
preprocess_config.subj=subj;
preprocess_config=config_preprocess(preprocess_config);

inspect_config=[];
inspect_config.show_biosig=true;
inspect_config=config_inspect(inspect_config);
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
EEG=pop_biosig(preprocess_config.paths.bdffile,preprocess_config.opts{:});
% detrend data
if inspect_config.detrend
    data_=detrend(EEG.data');
    EEG.data=data_';
    clear data_
end

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
        'detrend',true ...
        );
    fields=fieldnames(defaults);
    for ff=1:numel(fields)
        if ~isfield(config,fields{ff})||isempty(config.(fields{ff}))
            config.(fields{ff})=defaults.(fields{ff});
        end
    end
end