% script for estimating gamma distribution params that fit our stimuli
% where now we're using a Beta distribution instead since realizing the
% upper/lower bounds of the distribution are sortof important

%TODO: toggle var for normalizing mod spectra before plotting or not
%TODO: toggle switch for plotting peakrates using Ding envelopes or
%TODO: figure out number of frequency bands calculated in the ding
%spectrogram
%broadband
% assumes root dir is stimuli
clear%, clc
logTicks=2.^(-2:5);
xlims=[logTicks(1), logTicks(end)+1];
modSpecDir="./modSpectra";
peakRateDir="./peakRate";
norm=false;
% modSpecFile='../stimuli/regRandWrinkleModspectra.mat';
% modSpecFile='../stimuli/randEyeballWrinkleModspectra.mat';
% modSpecFile='../stimuli/randEyeballWrinkleModspectra.mat';

% NOTE: use longPauses file because in the getPeakrate script we exclude
% intervals based silTol - peakIndx is preserved though so could recalculate
% them from those numbers if wanted to
ogPeakRateFile=sprintf('%s/wrinklePeakrateLongpauses.mat',peakRateDir);
% peakRateFile='wrinkleRegRandPeakrate.mat';
% peakRateFile='wrinkleOgPeakratemaxInt.mat';

% use these as cutoffs since actually occur where we are censoring the
% output gamma distribution, not the silence tolarance for segment
% detection step - then just use the peakRate intervals with long pauses
minInt = 0.125;
maxInt = 0.5;

% switch peakRateFile
%     case 'wrinklePeakrateLongpauses.mat'
%         % just to jerry rig the "truncated fit"
%         % maxInt shouldnt even be a thing in this file tbh
%         maxInt=max(cat(1,peakRateIntervals));
% end
% fitDist='freq'; % time or freq
fitMethod=2; % 1 (naive BETA pdf fit) or 2 (truncated likelihood)

timeUnits='s'; %seconds or milliseconds
addManualGamma=false;
switch timeUnits
    case 's'
        unitFactor=1;
        freqUnits='hz';
    case 'ms'
        unitFactor=1e3;
        freqUnits='khz';
end
tMax=maxInt*unitFactor;
tMin=minInt*unitFactor;

% intervals saved in seconds, but will convert to milliseconds to improve
% fit convergence (numerical instability error if all values are less than
ogIntervals=load(ogPeakRateFile,'peakRateIntervals');
ogIntervals=ogIntervals.peakRateIntervals;

ogIntervals=unitFactor*cat(1,ogIntervals{:});
ogIntervals(ogIntervals>tMax|ogIntervals<tMin)=[];
ogFreqs=1./ogIntervals;
% scale data to 0-1 
ogFreqsScaled=rescaleData(ogFreqs,1/tMin,1/tMax);

meanInterval=mean(ogIntervals);
meanFreq=mean(ogFreqs);
% estimate the PDF using kernel density estimation
% xi=linspace(0, max(ogIntervals)+.1, 100);
[F,xiPdf]= ksdensity(ogFreqs);
% x values for plotting fitted pdf
% NOTE: don't start from 0 since log(0)-> inf so fminsearch won't converge!
xiFit = linspace(1/tMax, 1/tMin, 100);
xiFitScaled=rescaleData(xiFit,1/tMin,1/tMax);
fprintf('Original peakrate mean interval/syllable rate: %.2g %s (%.2g %s)\n', ...
    meanInterval,timeUnits,meanFreq,freqUnits)
% fit a gamma distribution to data
switch fitMethod
    case 1
        % fit usign fitdist while ignoring any cutoff value in distribution
        pd = fitdist(ogFreqsScaled, 'beta');
        % NOTE: turns out you can call mean on pd objects and it knows how
        % to use the parameters to calculate the mean without explicit
        % coding
        fitMeanFreq=unscaleData(mean(pd),1/tMin,1/tMax);
        fitMeanInterval=1./(fitMeanFreq);
        fprintf('Naive Beta distribution fit a,b parameters: %.2g, %.2g -> mean freq: %.2g %s (%.2g %s)\n', ...
            pd.a,pd.b,fitMeanFreq,freqUnits,fitMeanInterval,timeUnits)

        
        % generate the fitted gamma PDF
        pdfValues = pdf(pd, xiFitScaled); % Beta PDF values
        pdfFitString='naive';
    case 2

        % use maximum likelhood estimate with normalization factor to
        % account for distribution cutoff
        %initial guess at a,b params, from fitting naive gamma in case 1
        switch timeUnits
            % note: ms almost definitely wrong - the ones in seconds are
            % from aarons fit to the frequency
            case 's'
                params0=[1.4, 1.2];
            case 'ms'
                % these havent been adjusted for kHz
                % params0=[6.7 23];
        end
        % truncatedGammaLikelihood = @(params) -sum(log(gampdf(xiFit, params(1), params(2)) ./ ...
        %                                      (1 - gamcdf(tMax, params(1), params(2)))));

        % truncatedGammaLikelihood = @(params) -sum(log(gampdf(ogFreqs, params(1), params(2)))) + ...
        %                                     numel(ogFreqs)*log(1-gamcdf(1./tMax,
        %                                     params(1), params(2)));\
        % added in an upper bound in truncated fit norm factor to match distribution better:
        betaLikelihood = @(params) -sum(log(betapdf(ogFreqsScaled, params(1), params(2)))); 
        options = optimset('Display','iter');
        optParams=fminsearch(betaLikelihood, params0,options);
        % fitMeanFreq=optParams(1)*optParams(2);
        % fitMeanInterval=1./fitMeanFreq;
        fprintf('Beta distribution fit a,b parameters: %.2g, %.2g -> ...\n', ...
            optParams(1),optParams(2))
        pdfValues=betapdf(xiFitScaled,optParams(1),optParams(2));
        % force values below cutoff to be zero:
        % pdfValues(xiFit<(1/tMax))=0;
        pdfFitString='not truncated';
        %NOTE: I think correction factor should be included in output cdf as well:
        % ./(1-gamcdf(tMax,optParams(1),optParams(2)))
end




figure, hold on
histogram(ogFreqs, xiPdf,'Normalization','pdf')
plot(xiPdf, F)
% Plot the fitted Beta distribution PDF
plot(xiFit, pdfValues);
xlabel(sprintf('Frequency (%s)',freqUnits))
% xlabel(sprintf('Time (%s)',timeUnits))
legend('histogram', 'pdf estimate',sprintf('%s fitted beta PDF', pdfFitString))
title('Beta distribution fit to peakRate frequencies')
hold off
%% Plot the truncated fit against the ones used to generate stimuli along with histogram
% truncated fit a,b = 3.6, 1.9 already evaluated in above cell
% eyeball fit a,b=6.3,1.05
%TODO: fix normalization term here
eyeParams=[6.3 1.05];
pdfEyeball=gampdf(xiFit,eyeParams(1),eyeParams(2))./(gamcdf(1/tMin,eyeParams(1),eyeParams(2))-gamcdf(1/tMax,eyeParams(1),eyeParams(2)));
pdfEyeball(xiFit<1/tMax|xiFit>1/tMin)=0;
% eyeball "broadened" fit a,b=3.7,2
eyebroadParams=[3.7 2];
% just divided product of optimized params a,b (6.6342) by 3 to make scale
% bigger and see what shape should be to preserve mean
% eyebroadParams=[2.2114 3];

pdfEyebroad=gampdf(xiFit,eyebroadParams(1),eyebroadParams(2))./(gamcdf(1/tMin,eyebroadParams(1),eyebroadParams(2))-gamcdf(1/tMax,eyebroadParams(1),eyebroadParams(2)));
pdfEyebroad(xiFit<1/tMax|xiFit>1/tMin)=0;
figure, hold on
histogram(ogFreqs, xiPdf,'Normalization','pdf','DisplayName','1/intervals')
% truncated fit, already evaluated in previous cell
plot(xiFit, pdfValues,'DisplayName','truncated MLE fit');
plot(xiFit,pdfEyeball,'DisplayName','Eyeball estimate fit');
plot(xiFit,pdfEyebroad,'DisplayName','Eyeball estimate broadened');
xlabel(sprintf('Frequency (%s)',freqUnits))
% xlabel(sprintf('Time (%s)',timeUnits))
title('Truncated MLE Gamma distribution fit vs eyeball estimates')
set(gca, 'XScale','log','Xtick',logTicks,'XLim',xlims)

legend()
hold off
%% load modulation spectra and normalize



ogModSpecFile=sprintf('./%s/ogWrinkleDingMS.mat',modSpecDir);
randModSpecFile=sprintf('%s/randWrinkleDingMS.mat',modSpecDir);
randSameModSpecFile=sprintf('%s/randSameWrinkleDingMS',modSpecDir);
randEyeballModSpecFile=sprintf('%s/randEyeballWrinkleDingMS',modSpecDir);

msFreqs=load(ogModSpecFile,'freqs');
msFreqs=msFreqs.freqs;
msFreqsDeltas=round(diff(msFreqs),5); 
if ~all(msFreqsDeltas==msFreqsDeltas(1))
    error('dude')
else
    deltaFreqs=msFreqsDeltas(1);
end
clear msFreqsDeltas
ogModSpectra=load(ogModSpecFile,'modSpectra');
ogModSpectra=ogModSpectra.modSpectra.og;
meanOgModSpec=mean(ogModSpectra,1);
clear ogModSpectra
randModSpectra=load(randModSpecFile,'modSpectra');
randModSpectra=randModSpectra.modSpectra.rand;
meanRandModSpec=mean(randModSpectra,1);
clear randModSpectra
randSameModSpec=load(randSameModSpecFile,'modSpectra');
randSameModSpec=randSameModSpec.modSpectra.randSame;
meanRandSameModSpec=mean(randSameModSpec,1);
clear randSameModSpec
randEyeballModSpec=load(randEyeballModSpecFile,'modSpectra');
randEyeballModSpec=randEyeballModSpec.modSpectra.randEyeball;
meanRandEyeballModSpec=mean(randEyeballModSpec,1);
clear randEyeballModSpec

if norm
    msYlabel='Normalized Amplitude (A.U.)';
    % note: assuming frequency spacing is uniform
    % meanOgModSpec=meanOgModSpec./(deltaFreqs.*trapz(meanOgModSpec));
    meanOgModSpec=normalizeByArea(meanOgModSpec,deltaFreqs);  
    % meanRandModSpec=meanRandModSpec./(deltaFreqs.*trapz(meanRandModSpec));
    meanRandModSpec=normalizeByArea(meanRandModSpec,deltaFreqs);
    % meanRandSameModSpec=meanRandSameModSpec./(deltaFreqs.*trapz(meanRandSameModSpec));
    meanRandSameModSpec=normalizeByArea(meanRandSameModSpec,deltaFreqs);
    meanRandEyeballModSpec=normalizeByArea(meanRandEyeballModSpec,deltaFreqs);
else
    msYlabel='Absolute Amplitude (idk units?)';
end


%% Plot modulation spectrum vs truncated MLE fit 


figure, hold on
plot(xiFit, pdfValues,'DisplayName','truncated fit gamma pdf');
plot(msFreqs,meanOgModSpec,'DisplayName','Og Wrinkle in time ms');
% plot(repmat(meanFreq,2,1), 0:1, 'DisplayName', 'mean(1/intervals)')

legend()
xlabel(sprintf('Frequency (%s)',freqUnits))
ylabel(msYlabel)
title('Og Wrinkle in Time vs peakRate pdf')
set(gca,'XScale','log','Xtick',logTicks,'XLim',xlims)
hold off
%% plot the rand mod spec with it's generating pdf
%TODO: only execute this when rand is available (or change the plotting
%options as appropriate)
%NOTE THIS WILL PROBABLY ONLY WORK WITH THE REGRANDPEAKRATEFILE SINCE
%maxInt DOES NOT HAVE IT AND NEITHER DOES LONG PAUSES (I THINK)


figure, hold on
plot(xiFit,pdfEyebroad,'DisplayName','Generating pdf for rand stimuli');
% plot(xiFit, pdfValues,'DisplayName','truncated MLE fit');
plot(msFreqs,meanRandModSpec,'DisplayName','Rand Wrinkle in time ms');
% plot(repmat(meanFreq,2,1), 0:1, 'DisplayName', 'mean(1/intervals)')
xlabel(sprintf('Frequency (%s)',freqUnits))
ylabel(msYlabel)
legend()
set(gca,'XScale','log','Xtick',logTicks,'XLim',xlims)
title('Rand stimuli mod spec vs generating pdf')
hold off
%% plot both og and rand together 

figure, hold on

plot(msFreqs,meanOgModSpec,'DisplayName','Og Wrinkle in time ms');
plot(xiFit, pdfValues,'DisplayName','truncated MLE fit for og');

plot(msFreqs,meanRandModSpec,'DisplayName','Rand Wrinkle in time ms');
plot(xiFit,pdfEyebroad,'DisplayName','Generating pdf for rand stimuli');

% plot(repmat(meanFreq,2,1), 0:1, 'DisplayName', 'mean(1/intervals)')

legend()
xlabel(sprintf('Frequency (%s)',freqUnits))
ylabel(msYlabel)
set(gca,'XScale','log','Xtick',logTicks,'XLim',xlims)
title('Rand stimuli mod spec vs generating pdf')
hold off
%% do calculations for comparing randSame mod spec w og and pdf (doesn't plot anything)

randSamePeakRateFile=sprintf('%s/wrinkleRandSameEyeballPeakrate',peakRateDir);

randSamePeakrate=load(randSamePeakRateFile,'peakRate','conditions','fs');
% note: order of three lines below is important (mainly the last one being
% last is what matters)
randSameFs=randSamePeakrate.fs;
randSameConditions=randSamePeakrate.conditions;
randSamePeakrate=randSamePeakrate.peakRate;
% note intervals for these truncate based on silTol so gotta re-calculate
% them from index
for cc=1:numel(randSameConditions)

    [tempIntervals,tempFreqs]=getPeakRateIntervals(randSamePeakrate(cc,:),randSameFs,tMin,tMax);
    randSameIntervals.(randSameConditions{cc})=unitFactor.*tempIntervals;
    randSameFreqs.(randSameConditions{cc})=tempFreqs./unitFactor;

    clear tempIntervals
end
fprintf('done computing intervals.\n')
%% Plot og pdf vs randSame peakrate intervals
%TODO: actually probably better to plot the histograms on separate but
%linked axes 
%TODO: add informative titles
figure, hold on
plot(xiFit, pdfValues,'DisplayName','truncated MLE fit for og','Color','red');
plot(xiFit,pdfEyebroad,'DisplayName','Rand generating pdf (eyeball estimate fit)','Color','blue');
for cc=1:numel(randSameConditions)
    lgndStr=sprintf('1/intervals: %s',randSameConditions{cc});
    histogram(1./randSameIntervals.(randSameConditions{cc}),xiFit,'Normalization','pdf','DisplayName',lgndStr);
    clear lgndStr
end

% histogram(ogFreqs, xiPdf,'Normalization','pdf','DisplayName','Og Stim 1/intervals')
legend()
set(gca, 'XScale','log','Xtick',logTicks,'XLim',xlims)
title('Current rand stimuli distribution narrower than scrambled but unaltered distribution?')
hold off

%%
%TODO: add eyeball fit pdf and modspec (need to load the modspec first)




figure, hold on
plot(msFreqs,meanOgModSpec,'DisplayName','OG Wrinkle in time ms');
plot(msFreqs,meanRandSameModSpec,'DisplayName','randSame')
plot(xiFit, pdfValues,'DisplayName','truncated MLE fit for og');
title('effect of scrambling syllables using og distribution on modulation spectrum?')

legend()
set(gca, 'XScale','log','Xtick',logTicks,'XLim',xlims)
hold off
%% compare mod spectra of rand against estimates of "unaltered" modspecra
% rand should be different from og, randsame, and randeye, all of which
% should be similar to each other - but it looks like 

figure, hold on

plot(msFreqs,meanOgModSpec,'DisplayName','Og ms');
% plot(xiFit, pdfValues,'DisplayName','truncated MLE fit for og');

plot(msFreqs,meanRandModSpec,'DisplayName','Rand ms');

plot(msFreqs,meanRandSameModSpec,'DisplayName','randSame ms');

plot(msFreqs,meanRandEyeballModSpec,'DisplayName','randEye')
% plot(xiFit,pdfEyebroad,'DisplayName','Generating pdf for rand stimuli');

% plot(repmat(meanFreq,2,1), 0:1, 'DisplayName', 'mean(1/intervals)')

legend()
xlabel(sprintf('Frequency (%s)',freqUnits))
ylabel(msYlabel)
set(gca,'XScale','log','Xtick',logTicks,'XLim',xlims)
title('randSame should look like og but looks like rand')
hold off


%% COMPARE PEAKFREQUENCY DISTRIBUTIONS USING WEIRD POWER ENVELOPE CALCULATION
% other file we want later:wrinkleRandSameRandEyeballPeakrateDing
%TODO: convert to structures instead of cells so we can get the conditions
%we want without having to look at conditions variable that's in the file
% temp=load(sprintf('%s/wrinkleRandOgPeakrateDing',peakRateDir));
% dingOgPeakrate=temp.peakRate(2,:);
% [dingOgIntervals,dingOgFreqs]=getPeakRateIntervals(dingOgPeakrate,temp.fs,tMin,tMax);
% % dingRandPeakrate=temp.peakRate(1,:);
% 
% 
% figure, hold on
% histogram(dingOgFreqs,'DisplayName','og peak frequencies using ding narrowband env','Normalization','pdf')
% % plot(xiFit,pdfValues,'DisplayName','pdf estimate using broadband envelope')
% legend()
% hold off

%% DEFINE HELPERS
function unscaledShit=unscaleData(scaledShit,uBound,lBound)
    %re-scale data to range [lBound,uBound] range
    unscaledShit=(uBound-lBound).*scaledShit + lBound;
end
function scaledShit=rescaleData(unscaledShit,uBound,lBound)
    arguments
        unscaledShit double
        uBound double = max(unscaledShit)
        lBound double = min(unscaledShit)
    end
    scaledShit=(unscaledShit-lBound)/(uBound-lBound);
end
function normalizedShit=normalizeByArea(unnormalizedShit,delta)
    normalizedShit=unnormalizedShit./(delta.*trapz(unnormalizedShit));
end
function [peakIntervals,peakFreqs]=getPeakRateIntervals(peakRate,fs,tMin,tMax)
    % [peakIntervals,peakFreqs]=getPeakRateIntervals(peakRate,fs,tMin,tMax)
    % intervals will be in seconds, freqs in 1/s
    peakIntervals=cellfun(@(x) x(:,1)./fs, peakRate,'UniformOutput',false);
    %TODO: make this a function to avoid copy-paste errors
    peakIntervals=diff(cat(1,peakIntervals{:}));

    peakIntervals(peakIntervals>tMax|peakIntervals<tMin)=[];
    peakFreqs=1./peakIntervals;
    % randSameIntervals.(randSameConditions{cc})=tempIntervals;

end