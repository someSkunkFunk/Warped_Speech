function [env, sgram] = extractGCEnvelope(audiofile,fs,down_factor,NumCh,FRange)

arguments
    audiofile char
    fs (1,1) double = 128
    down_factor (1,1) double = 5
    NumCh (1,1) double = 16
    FRange (2,1) double = [80 8020]
end

[audio,wav_fs]=audioread(audiofile);
audio = audio(:,1);


%% Filter setup
% Lowpass filter
Fpass = 1.8e4;
Fstop = 2.2e4;
Apass = 1;
Astop = 60;
h = fdesign.lowpass(Fpass,Fstop,Apass,Astop,wav_fs);
lpf = design(h,'cheby2','MatchExactly','stopband');
clear Fpass Fstop Fs h;

% GammacHirp filterbank
GCparam.fs = wav_fs/down_factor;
GCparam.FRange = FRange;
GCparam.OutMidCrct = 'ELC';
GCparam.NumCh = NumCh;
% GCparam.OutMidCrct = 'No';
% GCparam.Ctrl = 'dyn';

%% Envelope extraction
% Filter below Nyquist frequency
audio = filtfilthd(lpf,audio);

% Downsample Before to reduce computation time
audio= nt_dsample(audio,wav_fs/GCparam.fs);

sgram = GCFBv210(audio',GCparam);

% Calculate narrowband and broadband envelopes
for chn=1:size(sgram,1)
    sgram(chn,:)=abs(hilbert(sgram(chn,:)));
end

env = mean(sgram,1);

env=resample(env',fs,GCparam.fs);
    
sgram = resample(sgram',fs,GCparam.fs);

