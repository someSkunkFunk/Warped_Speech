function SiN = noisySpeech(S,Fs,SNR,speech_delay)

% Things that need tweaking
% 1. add ramp to noise
ramp_dur = min(1,abs(speech_delay))*Fs; % seconds
target_db = 70;
noise_spec = 1; % 1 = speech shaped, 2 = flat

if ~exist('speech_delay','var')
    speech_delay = 0 ;
end

% Initialize sound level meter
sm = splMeter('TimeWeighting', 'Fast','FrequencyWeighting', 'A-weighting','SampleRate', Fs);

% Silence contaminates the reading. We must remove.
dur = 0.15; lvl = 0.01; % Hardcoded silence level and duration thresholds
S_noSil = removeSilence(S,Fs,lvl,dur);

% Measure sound level
mean(sm(S_noSil)); % First reading is botched
SPL = mean(sm(S_noSil));

switch sign(speech_delay)
    case {0,1}
        % Add silence before speech to add delay
        S = cat(1,zeros(Fs.*speech_delay,1),S);
        
        % Generate some noise
        N = makeNoise(S,noise_spec);

        % Measure starting SNR
        pre_SNR = SPL-mean(sm(N));
    case -1
        % Generate some noise
        N = makeNoise(S,noise_spec);

        % Measure starting SNR
        pre_SNR = SPL-mean(sm(N));

        % Silence the beginning to delay the noise
        onramp = zeros(abs(speech_delay).*Fs,1);
        onramp(end-ramp_dur:end) = linspace(0,1,ramp_dur+1);

        N(1:abs(speech_delay.*Fs)) = N(1:abs(speech_delay.*Fs)).*onramp;
end

% Adjust noise level
N = N.*db2mag(pre_SNR-SNR);

% Add S and N and scale overall level
SiN = S+N;
pre_db = mean(sm(SiN));
SiN = SiN.*db2mag(target_db-pre_db);
end



function wf_noSil = removeSilence(wf,fs,thr,dur)
S=regionprops(wf<thr,'PIxelIdxList','Area');
I = find([S.Area]>fs*dur);
wf_noSil = wf;
for ii = length(I):-1:1
    wf_noSil(S(I(ii)).PixelIdxList)=[];
end
wf_noSil(wf_noSil==0)=[];
end

function N = makeNoise(S,noise_spec)
switch noise_spec
    case 1 % Speech Shaped
        N = real(ifft(abs(fft(S)).*exp(2i*pi*rand(size(S)))));
    case 2 % flat
        N = randn(size(S));
    otherwise     
end
end