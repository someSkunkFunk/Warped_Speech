%clip long silences from wrinkle in time
% first, load peakrate with long pauses included
clear, clc

%TODO: plot waveforms before/after clipping long pauses for a visual
userProfile=getenv('USERPROFILE');
peakRateFile="wrinklePeakrateLongpauses.mat";
load(peakRateFile) %NOTE: not sure if using peakrate times will be as useful... 
% unless we pair it with an additional env threshold to ensure we're not
% removing additional sounds
% locate original wav file for each stimulus/condition
ogStimFolder=sprintf('../stimuli/%s/og/',stimset);
clippedStimFolder=sprintf('../stimuli/%s/noSil/',stimset); %note: they won't actually have clicks so kinda stupid to put in with clicks
D=dir([ogStimFolder '*.wav']);
soundChn=1;
envThresh=0.005;
%%
for ss=1:numel(D)
    fprintf('stim %d of %d...\n',ss,numel(D))
    fnm=D(ss).name;
    [wf, fs]=audioread([ogStimFolder fnm]);
    % lpIdx=find(peakRateIntervals{ss}>silToll);
    %note: realizing that if we just clip stuff between peaks, we are also
    %clippling part of the waveform for each sound segment since peakrate
    %is peak in derivative....
    env=abs(hilbert(wf(:,soundChn)));
    wfClipped=wf(env>envThresh,soundChn);
    audiowrite([clippedStimFolder fnm], wfClipped,fs);

end
% remove samples occuring between intervals longer than silTol

% save new ones somewhere

% calculate new mod spec (on the mod spec calculating script - not here)