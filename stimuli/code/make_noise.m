% make white nosie
clear
noise_dir='./noise_test/';
% fs=44100;
fs=20000; % for speedier computation while preserving main spectral properties of speech
dur=64; %seconds
A=1.0; % amplitude - should we modify this?
lpf_noise=1.0e3; %make negative or zero to do nothing

if ~exist(noise_dir,'dir'), mkdir(noise_dir); end
fnm=sprintf('%s/pure_noise/white_noise.wav',noise_dir);
if ~exist(fnm,'file')
    noise_wav=randn(dur*fs,1);
    %normalize to 1 then force amplitude to be what we want
    noise_wav=A.*(noise_wav./max(abs(noise_wav)));
    
    audiowrite(fnm,noise_wav,fs);
else
    [noise_wav, fs_noise]=audioread(fnm);
    if sign(lpf_noise)==1
        % lowpass the noise
        fprintf('low-passing noise at %0.3g Hz\n',lpf_noise)
        Hd=getLPFilt(fs,lpf_noise);
        noise_wav=filtfilthd(Hd,noise_wav);
        %normalize to 1 again and reset A to original val again
        noise_wav=A.*(noise_wav./max(abs(noise_wav)));
    end
end

figure
plot(0:1/fs:dur-1/fs,noise_wav,'DisplayName','white noise')
legend()

% speech-envelope modulated noise
% TODO: plot an example speech waveform next to corresponding speechy noise
% TODO: make this portion of script run with low-passed white noise
speech_dir='./wrinkle_wClicks/og';
D=dir([speech_dir '/*.wav']);
click_ch=2;

for dd=1:numel(D)
    fprintf('making speechy noise %d\n',dd)
    % get speech wav
    audio_path=[speech_dir '/' D(dd).name];
    if dd==1
        [sp_wav,sp_fs]=audioread(audio_path);
        [p, q] = rat(fs/sp_fs);
    else
        [sp_wav,~]=audioread(audio_path);
    end
    % remove click
    %TODO: figure out why clicks seem to be missing now....???
    % weird... on urmc computer the clicks are still there....
    sp_wav(:,click_ch)=[];
    % get envelope
    env=abs(hilbert(sp_wav));
    % downsample for speed
    env=resample(env,p,q);
    % assuming duration of all speech stims is the same here
    % rectify and normalize
    env(env<0)=0;
    env=env./max(env);
    % modulate with speech envelope
    sp_noise=env.*noise_wav;
    % % save output
    if sign(lpf_noise)==1
        out_fnm=sprintf('%sspeechy_noise_lpf%0.3g/speechy_noise%03d.wav',noise_dir,lpf_noise,dd);
    else
        out_fnm=sprintf('%sspeechy_noise/speechy_noise%03d.wav',noise_dir,dd);
    end
    if ~exist(fileparts(out_fnm),'dir'), mkdir(fileparts(out_fnm)); end
    audiowrite(out_fnm,sp_noise,fs);
end
disp('noisy speech made.')
%% plot example speech-modulated noise waveform alongisde speech

sp_fnm='./wrinkle_wClicks/og/wrinkle010.wav';
[sp_wav_ex,fs_sp]=audioread(sp_fnm);
ns_fnm='./noise_test/speechy_noise_lpf8.5e+03/speechy_noise010.wav';
[sp_noise_ex,fs_nsp]=audioread(out_fnm);
speech_t=0:1/sp_fs:dur-1/sp_fs;
noise_t=0:1/fs:dur-1/fs;
figure
ax(1)=subplot(2,1,1);
plot(speech_t,sp_wav_ex,'DisplayName','speech')
ax(2)=subplot(2,1,2);
plot(noise_t,sp_noise_ex,'DisplayName','moded noise')
linkaxes(ax,'x')
xlabel('time (s)')

%% plot FFT of speech stimuli to determine min sample rate
example_fnm='./wrinkle_wClicks/og/wrinkle001.wav';
[ex_speech,fs_sp]=audioread(example_fnm);
%remove click
click_ch=2;
ex_speech(:,click_ch)=[];
% get fft and plot
ex_speech_fft=fftshift(fft(ex_speech));
% get power spectrum instead
ex_speech_pow=abs(ex_speech_fft).^2/numel(ex_speech_fft);
fft_freqs=(-numel(ex_speech)/2:numel(ex_speech)/2-1)*(fs_sp/numel(ex_speech));
figure
plot(fft_freqs,ex_speech_pow)
xlabel('Hz')
title('example speech power spec')
% % % by eyeball, I think a nyquist of 10kHz should be fine - 20 kHz sample
% % rate for analysis
% %TODO: another number we need is the frequency where speech spectral
% %content becomes "negligible" so we can low-pass filter our noise below 10
% kHz until we reach a value matching speech...