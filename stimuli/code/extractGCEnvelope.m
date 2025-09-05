function [env, sgram] = extractGCEnvelope(audiofile,fs,down_factor,NumCh,FRange)
% [env, sgram] = extractGCEnvelope(audiofile,fs,down_factor,NumCh,FRange)
% 9/5/2025 - extended functionality to accept already read file waveforms as input
arguments
    audiofile
    fs (1,1) double = 128
    down_factor (1,1) double = 5
    NumCh (1,1) double = 16
    FRange (2,1) double = [80 8020]
end

switch class(audiofile)
    case 'char'
        [audio,wav_fs]=audioread(audiofile);
    case 'struct'
        if isequal(sort(fieldnames(audiofile)),sort({'fs';'wf'}))
            audio=audiofile.wf;
            wav_fs=audiofile.fs;
        else
            disp(audiofile)
            error('^^^audiofile struct incorrect^^^')
        end
    otherwise
        error('audiofile should be either a struct containing wf and fs or file path.')
end
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

