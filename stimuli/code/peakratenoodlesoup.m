% incomplete script for visualizing peakrate events on a particular
% stimulus instance
clear, clc
fastSlowPeakrateFile="wrinkleFastSlowPeakrate.mat";
load(fastSlowPeakrateFile)
clear plotPeakRateAmplitude peakRateIntervals peakTimes
% todo: clear those extra vars from original file
% choose example stimulus condition, wav index
plotcc=1;
plotwindx=1;
% get peakRate for chosen stimulus
plotPeakrate=peakRate{plotcc,plotwindx};
plotPeakRateIndx=plotPeakrate(:,1);
plotPeakRateAmplitude=plotPeakrate(:,2);

% load example wav:
% stimset='wrinkle_wClicks';
% stimulifolder='../stimuli/wrinkle_wClicks';
switch round(scalecond(plotcc),2)
    case 1
        wavspath=sprintf('%s/og',stimulifolder);
    case {.67, 1.5}
        wavspath=sprintf('%s/%0.2f',stimulifolder,scalecond(plotcc));
end
D=dir(sprintf('%s/wrinkle*.wav',wavspath));
fprintf('starting condition: %0.2f\n', scalecond(plotcc));

fprintf('using stim wav %d...\n',plotwindx)
flpth=sprintf('%s/%s',wavspath,D(plotwindx).name);
[X,fs]=audioread(flpth);
X=X(:,1);
 % Extract smoothed envelope
if plotwindx==1
    Hd=getLPFilt(fs,fc);
end
env=abs(hilbert(X));
env=filtfilthd(Hd,env);
env(env<0)=0;

% differentiate
denv=zeros(length(env), 1);
denv(2:end,:)=diff(env);

% rectify

denv(denv<0) = 0; 


figure
seglims=[316025 332811]; % this was the "in an old" for fast speech
denvscale=1000;
plot(X)
hold on
plot(denvscale.*denv)
scatter(plotPeakRateIndx,denvscale.*plotPeakRateAmplitude)
plot(env)
legend('x', 'denv','peakrate','env')
title(sprintf('In an old - fast wrinkle001 using silTol=%0.3f',silTol))
xlim(seglims)
hold off
% figure
% plot(X)
% hold on
% plot(denv)
% scatter(peakIndx(dum),denv(dum))
% plot(env)
% legend('x', 'denv','peakrate','env')