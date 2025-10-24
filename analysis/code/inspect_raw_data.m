function inspect_raw_data(subj,config)
    %NOTES: hard to decode without detrending first
    % could also help to use regular matlab plot but dont do for all chans
    % at once cuz matlab will freak
    defaults=struct('chns', ...
        1:128);
    preprocess_config.subj=subj;
    preprocess_config=config_preprocess(preprocess_config);

    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    EEG=pop_biosig(preprocess_config.paths.bdffile,preprocess_config.opts{:});
    [ALLEEG, EEG, CURRENTSET]=eeg_store([],EEG,1);
    pop_eegplot( EEG, 1, 1, 1);
    eeglab redraw
    % plot timeseries

    % if 
    % plot spectra
end