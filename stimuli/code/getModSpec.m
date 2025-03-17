% plot avg modulation spectra for reg and irreg and og wrinkle in time
% stimuli
%TODO: MAKE NAME OF OUTPUT FILE DEPEND ON CONDITIONS
%TODO: COMBINE EYEBALL RAND AND MLE TRUNCATED FIT RAND INTO ONE FILE FOR
%COMPARATIVE PLOTTING IN OTHER SCRIPT
clear
clc
dependencies_path=('../../dependencies/');
addpath(genpath(dependencies_path));
% note we're not using local box folder here which may cuase issues down
% the line, but we're trusting the wav files on shared box folder not to
% change except by my own decision (and for me to keep track of when that
% happens)
user_profile=getenv('USERPROFILE');
% note assuming stimuli without clicks held in shared box folder
% stimuli_dir=sprintf('%s/Box/Lalor Lab Box/Research Projects/Aaron - Warped Speech/stimuli',user_profile);
% stimuli_dir='./';
stimuli_dir=sprintf('%s/Box/my box/LALOR LAB/oscillations project/MATLAB/Warped Speech/stimuli/rewarped',user_profile);

% stimset='wrinkle';
% stimset='wrinkle_wClicks';
stimset='wrinkle';
conditions={'randSame'};
% conditions={'randEyeball'};
% conditions={'og','reg','rand'};
nfreqs=48000;%note not sure how to determine this value for all the conditions;
% i think it depends on number of samples basically; but also i think we
% truncated in get_avg_ms since higher frequencies not necessary
% modSpectraFile="noSilWrinkleModspectra.mat";
% modSpectraFile="regRandWrinkleModspectra.mat"

modSpectraFile="randSameWrinkleModspectra.mat";
if ~exist(modSpectraFile,'file')
    for cc=1:numel(conditions)
        cond=conditions{cc};
        cond_dir=sprintf('%s/%s/%s/',stimuli_dir,stimset,cond);
        d=dir([cond_dir '*.wav']);
        tempMS=zeros(numel(d),nfreqs);
        for ss=1:numel(d)
            fnm=d(ss).name;
            flpth=fullfile(cond_dir,fnm);
            % downsamples to 16kHz by default
            [ms, freqs]=get_avg_ms(flpth);
            tempMS(ss,:)=ms';
            clear fnm ms
        end
        modSpectra.(cond)=tempMS;
        clear tempMS 
    end
    save(modSpectraFile)
else
    load(modSpectraFile)
end
%%
% plot figure
logTicks=2.^(-2:5);
xlims=[logTicks(1), logTicks(end)+1];
for cc=1:numel(conditions)
    cond=conditions{cc};
    h=plot(freqs,mean(modSpectra.(cond),1));
    title('Average Mod Spectra')
    xlabel('frequencies')
    xlim([0 30])
    hold on
      
end
legend(conditions)
set(gca, 'Xscale','log','XTick',logTicks)
hold off
% savefig('noSilWrinkleMS')
        
% add mean frequency lines for each stim category

for cc=1:numel(conditions)
    cond=conditions{cc};
    tempMean=mean(modSpectra.(cond),1);
    condMeanFreq.(cond)=(tempMean*freqs)/nfreqs;
    [~, condMaxFreqIndx]=max(tempMean);
    condMaxFreq.(cond)=freqs(condMaxFreqIndx);
    fprintf('mean freq for %s: %0.2f Hz\n', cond,condMeanFreq.(cond))
    fprintf('max freq for %s: %0.2f Hz\n', cond,condMaxFreq.(cond))
    
    % plot([condMean condMean],[0 1]);
    % legend(sprintf('%s mean freq',cond))
end
