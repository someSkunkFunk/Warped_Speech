function [wf_warp,s] = rhythmicOldMan(wf,fs,k,peak_tol,sil_tol)
% k: 1->make regular, -1-> more irregular, 0-> shuffle randomly
% natural speech
% regularized speech intervals
% irregularized speech intervals
% - selecting from distribution
% - adding jitter (correlated or uncorrelated)
% shuffled speech

% dealing with artifacts
%  - look at modulation spectrum before and after
% behavioral experiment

% do modulation spectrum instead of peak rate intervals
% -compare Nai spectrum with Old Man spectrum


arguments
    wf double
    fs (1,1) double
    k (1,1) double = 1;
    peak_tol (1,1) double = 0.1;
    sil_tol (1,1) double = 0.3;
end

% env derivative peak rate (measured)
%TODO: make this equal to value from stimulus
f = 6.25;

% For repeatability
rng(1);

%% HI ANDRE 250
% We need TSM toolbox
addpath(sprintf('%s/TSM Toolbox',boxfolder))

% Extract envelope
Hd = getLPFilt(fs,10); %% Maybe don't filter so harshly?
env = abs(hilbert(wf));
env = filtfilthd(Hd,env);

% Find onsets
env_onset = diff(env);
env_onset(env_onset<0) = 0;

[~,Ifrom,~,p] = findpeaks(env_onset,fs);

% normalize prominence
p = p./std(p);

% Eliminate small peaks
Ifrom(p<peak_tol)=[];
p(p<peak_tol)=[];

seg = [[1; find(diff(Ifrom)>sil_tol)+1] [find(diff(Ifrom)>sil_tol); length(Ifrom)]];


% inter-segment interval
% updated each segment iteration to preserve timing between segments
ISI = 0;
[ItoReg,ItoShuff,ItoRand] = deal(zeros(size(Ifrom)));
for ss = 1:size(seg,1)
    % Get the intervals for the current segment
    IPI = diff(Ifrom(seg(ss,1):seg(ss,2)));
    
    % decide regularity
    switch sign(k)
        case -1 % Make more irregular
            if numel(IPI)>1 % if at least 3 peaks (2 intervals)
                nsamps = diff(seg(ss,:));
                secs = Ifrom(seg(ss,:));
                I = gammaRandInterval(nsamps,secs);
                ItoRand(seg(ss,1):seg(ss,2)) = I;
            else % else just use the original peaks
                ItoRand(seg(ss,1):seg(ss,2)) = Ifrom(seg(ss,1):seg(ss,2));
            end

        case 1 % Make more regular
%             ItoReg(seg(ss,1):seg(ss,2)) = linspace(Ifrom(seg(ss,1)),Ifrom(seg(ss,2)),diff(seg(ss,:))+1);

            if ss>1
                ISI = Ifrom(seg(ss,1))-Ifrom(seg(ss,1)-1);
                start_t = ItoReg(find(ItoReg,1,'last'))+ISI;
            else
                start_t = Ifrom(seg(ss,1))+ISI;
            end

            ItoReg(seg(ss,1):seg(ss,2)) = start_t+(0:diff(seg(ss,:)))./f;
            
        case 0 % Shuffle intervals
            ItoShuff(seg(ss,1):seg(ss,2)) = Ifrom(seg(ss,1))+[0; cumsum(IPI(randperm(length(IPI))))];
        
        otherwise
            if isnan(k)
                ItoReg(seg(ss,1):seg(ss,2)) = linspace(Ifrom(seg(ss,1)),Ifrom(seg(ss,2)),diff(seg(ss,:))+1);
            else
                error('wrong')
            end
    end
    % fprintf('%d\n',ii)

    

    % ItoRand = Ifrom+rand_jitter.*randn(size(Ifrom));
    % while min(diff(ItoRand))<0.02
    % ItoRand = Ifrom+rand_jitter.*randn(size(Ifrom));
    % end

    % Ole's idea: define each word's random interval by finding midpoint between words?

    % Ed's idea: Add jitter as a percentage of original interval.

end


switch sign(k)
    case -1
        Ito = Ifrom+(ItoRand-Ifrom).*abs(k);
    case 1
        Ito = Ifrom+(ItoReg-Ifrom).*abs(k);
    case 0
        Ito = ItoShuff;
    otherwise
        Ito = Ifrom+(ItoReg-Ifrom);
end

s = round([Ifrom Ito]*fs);
s = [ones(1,2); s; ones(1,2)*length(wf)];
end_sil = diff(s(:,1)); end_sil = end_sil(end);
s(end,2) = s(end-1,2)+end_sil;

param.tolerance = 256;
param.synHop = 256;
param.win = win(1024,2); % hann window
wf_warp = wsolaTSM(wf,s,param);

% Re-mixed a re-mix and it was back to normal.
if isnan(k)
    s = fliplr(s);
    wf_warp = wsolaTSM(wf_warp,s,param);
end

% Plotting
if 0
    env_fs = 2048; 
    env_warp = abs(hilbert(wf_warp));
    env_warp = filtfilthd(Hd,env_warp);

    env = resample(env,env_fs,fs);
    env_warp = resample(env_warp,env_fs,fs);

    t_env = linspace(0,length(env)./env_fs,length(env));

    figure
    ex(1) = subplot(2,1,1);
    plot(t_env,env./std(env))


    ex(2) = subplot(2,1,2);
    plot(t_env,env_warp./std(env_warp))
    linkaxes(ex,'x')%
end

end

function I = quantizedRand(seg,Q,Ifrom,wav_fs)

samps = round(Ifrom(seg(ss,1))*wav_fs):Q*wav_fs:round(Ifrom(seg(ss,2))*wav_fs);
I = sort(datasample(samps,diff(seg(ss,:))+1,'Replace',false));
I([1 end]) = round(Ifrom(seg(ss,:))*wav_fs);

while min(diff(I))>(0.04*wav_fs)&&max(diff(I))<(0.3*wav_fs)
    I = sort(datasample(samps,diff(seg(ss,:))+1,'Replace',false));
    I([1 end]) = round(Ifrom(seg(ss,:))*wav_fs);
end
I = I./wav_fs;
end


function [I,ii] = gammaRandInterval(nsamps,secs,a,b)

arguments
    nsamps (1,1) double
    secs (2,1) double
    a = 3.7;
    b = 2;
end

buff = 0.04;
minInt = 0.05;
maxInt = 0.5;
Icrit = 100000;

% Original distribution
% a=6.3;
% b=1.05;

% broader distribution
% a = 3.7;
% b = 2;

ii = 0;
I = 1./gamrnd(a,b,nsamps,1);
% are the intervales longer or shorter than we need and are any of them
% within the censored region [0,minINT] or [maxINT,Inf].
while sum(I)>diff(secs) || sum(I)<diff(secs)-buff || min(I)<minInt || max(I)>maxInt
    I = 1./gamrnd(a,b,nsamps,1); ii = ii +1;

    % if ~mod(ii,Icrit)
    %     fprintf('%d ',ii/Icrit)
    %     minInt = minInt-0.01;
    % end
end

% Create the timepoints from intervals and set the segment onset
I =  [0; cumsum(I)]+secs(1);
% Fix the last value to be perfect
I(end) = secs(2);

end