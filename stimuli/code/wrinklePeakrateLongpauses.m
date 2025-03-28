% wrink
clear, clc
%watch out for harcoded condition indices...
stimset='wrinkle_wClicks';
stimulifolder='../stimuli/wrinkle_wClicks';
fc=10; % cutoff frequency in Hz for envelope smoothing
condNms={'og'};
% set threshold for peaks to ignore based on smoothed envelope value
env_thresh = 0.005; % adjusted based on visual height of emplitdue between speech segments in 2/3 stimuli
nwavs=120; % per condition
nconditions=numel(condNms);
peakRate=cell(nconditions,nwavs);
peakRateIntervals=cell(nconditions,nwavs);
% silTol=0.3; % time in seconds for minimum silence duration between peakrates
peakTol=0.1;
peakRateFile="wrinklePeakrateLongpauses.mat";

if ~exist(peakRateFile,'file')


    wavspath=sprintf('%s/og',stimulifolder);
    
    D=dir(sprintf('%s/wrinkle*.wav',wavspath));

    for windx=1:length(D)
        fprintf('wav %d...\n',windx)
        flpth=sprintf('%s/%s',wavspath,D(windx).name);
        [X,fs]=audioread(flpth);

        % throw away clicks channel
        X=X(:,1);

        % Extract smoothed envelope
        if windx==1
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

        %note: thresholding does make significant difference

        % find peaks in env rate

        [~,peakIndx]=findpeaks(denv);
        [~,peakTimes,~,p]=findpeaks(denv,fs);

        % normalize prominence

        p = p./std(p);

        % Eliminate small peaks

        peakIndx(p<peakTol)=[];
        peakTimes(p<peakTol)=[];
        p(p<peakTol)=[];

        % find silent segments

        % seg = [[1; find(diff(peakTimes)>silTol)+1] [find(diff(peakTimes)>silTol); length(peakTimes)]];

        % interPeakIntervals=nan(numel(peakTimes)-numel(seg)+1,1);

        interPeakIntervals=diff(peakTimes);

        % for ss=1:size(seg,1)
        %     interPeakIntervals=[interPeakIntervals; diff(peakTimes(seg(ss,1):seg(ss,2)))];
        % end
        % clear ss

        % remove peaks where envelope is tiny
        % mask=zeros(size(X));
        % mask(peakIndx)=1;
        % mask(env<env_thresh)=0;
        % remove peaks that happen too quickly, NOTE: nvm that doesn't work
        % mask(find(diff(peakTimes)<silTol)+1)=0;     
        %TODO: check if this line above actually doing anything because
        %peaks not being filtered properly!
        % peakRateIndx=find(mask);
        % peakRateTimes=find(mask,fs);
        peakRateAmplitude=denv(peakIndx);
        peakRate{1,windx}=[peakIndx,peakRateAmplitude];
        peakRateIntervals{1,windx}=interPeakIntervals;
        
        % also get peakrate including long pauses
        clear mask X x denv peakIndx smooth_env 
        clear flpth seg interPeakIntervals p 
        clear interPeakIntervals peakRateAmplitude peakIndx
    end
    clear D wavspath windx
    save(peakRateFile, 'peakRate', 'peakRateIntervals','fs', ...
        'condNms','env_thresh', ...
        'fc','peakTol','silTol','stimset','Hd');
else
    load(peakRateFile)
end

nbins=100;
figure
cc=1;
ccIntervals=cat(1,peakRateIntervals{cc,:});
% ax(cc)=subplot(nconditions,1,cc);
histogram((1e3)*ccIntervals,nbins);
title(sprintf('og Wrinkle interpeak rate intervals with long pauses included'))
xlabel('ms')
ylabel('counts')
mean_interval=mean(ccIntervals);
mean_ms=(1e3)*mean_interval;
mean_syllable_rate=1/mean_interval;
fprintf('mean interval: %0.2f ms\n',mean_ms)
fprintf('mean peakRate rate: %0.2f (1/s)\n',mean_syllable_rate)
clear mean_interval mean_ms mean_syllable_rate ccIntervals cc

% linkaxes(ax,'x')


