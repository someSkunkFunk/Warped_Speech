clear;
clc;
addpath(sprintf('%s/TSM Toolbox',boxfolder))

stimfolder = 'C:/Users/aaron/Box/Projects/Warped Speech/stimuli/';
overwrite = 0;
% stimgroup = {'leagues','oldman','wrinkle'};
stimgroup = {'wrinkle'};
stimspeed = [0.5 2/3 3/2 2];
param.tolerance = 256;
param.synHop = 256;
param.win = win(1024,2); % hann window
for gg = 1:length(stimgroup)
    fprintf('processing stimulus: %s\n',stimgroup{gg})

    d = dir(sprintf('%s%s/og/*.wav',stimfolder,stimgroup{gg}));
    for dd = 1:length(d)
        fprintf('file: %s ',d(dd).name)
        audiofile = sprintf('%s%s/og/%s',stimfolder,stimgroup{gg},d(dd).name);
        [audio,wav_fs]=audioread(audiofile);

        % Warp Speech, natural cadence
        for ss = 1:length(stimspeed)
            savefile = sprintf('%s%s/%0.2f/%s',stimfolder,stimgroup{gg},stimspeed(ss),d(dd).name);
            if ~exist(savefile,'file') || overwrite

                fprintf('%0.2fx ',stimspeed(ss))
                audio_warp = wsolaTSM(audio,stimspeed(ss),param);

                if ~exist(sprintf('%s%s/%0.2f/',stimfolder,stimgroup{gg},stimspeed(ss)),'dir')
                    mkdir(sprintf('%s%s/',stimfolder,stimgroup{gg}),sprintf('%0.2f',stimspeed(ss)))
                end

                audiowrite(savefile,audio_warp,wav_fs);
            end
        end

        % Natural Speed, Warp Candence
        
        % Make more regular
        type = 'reg';
        savefile = sprintf('%s%s/%s/%s',stimfolder,stimgroup{gg},type,d(dd).name);
        
        if ~exist(savefile,'file') || overwrite
            fprintf('%s ',type)
            audio_warp = rhythmicOldMan(audio,wav_fs,1); 

            if ~exist(sprintf('%s%s/%s/',stimfolder,stimgroup{gg},type),'dir')
                mkdir(sprintf('%s%s/',stimfolder,stimgroup{gg}),type)
            end

            audiowrite(savefile,audio_warp,wav_fs);
        end

        % Make more random
        type = 'rand';
        savefile = sprintf('%s%s/%s/%s',stimfolder,stimgroup{gg},type,d(dd).name);
        
        if ~exist(savefile,'file') || overwrite
            fprintf('%s ',type)
            audio_warp = rhythmicOldMan(audio,wav_fs,-1); 

            if ~exist(sprintf('%s%s/%s/',stimfolder,stimgroup{gg},type),'dir')
                mkdir(sprintf('%s%s/',stimfolder,stimgroup{gg}),type)
            end

            audiowrite(savefile,audio_warp,wav_fs);
        end

        fprintf('\n')
    end
end

