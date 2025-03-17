clear;
clc;

userprofile=getenv('USERPROFILE');
stimfolder = sprintf('%s/Box/Lalor Lab Box/Research Projects/Aaron - Warped Speech/stimuli/',userprofile);
overwrite = 0;
filterbanks={'gc','bark'};
% stimgroup = {'leagues','oldman','wrinkle'};
stimgroup = 'wrinkle';
ntrials = 120;
% stimscale = [2/3 1 3/2];
stimscale=[1];
% fs = 128;
fs=44100;
outputFile=sprintf('WrinkleEnvelopes%dhz.mat',fs);
[env,sgram]=deal(cell(numel(filterbankslength(stimscale),ntrials));
for ss = 1:length(stimscale)
    for tt = 1:ntrials
        fprintf('**********************\n')
        fprintf('Speed = %0.2g, trial %d\n',stimscale(ss),tt)
        fprintf('**********************\n')
        if stimscale(ss)==1
            audiofile = sprintf('%s%s/og/wrinkle%0.3d.wav',stimfolder,stimgroup,tt);
        else
            audiofile = sprintf('%s%s/%0.2f/wrinkle%0.3d.wav',stimfolder,stimgroup,stimscale(ss),tt);
        end
        
        [env{ss,tt}, sgram{ss,tt}] = extractGCEnvelope(audiofile,fs);
    end
end
save(outputFile,'env','sgram','fs')