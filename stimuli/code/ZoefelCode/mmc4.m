
function [speechnoise] = construct_stimuli_without_spec_fluc(wavfile, freq_down, freq_up, n_bins, n_iter)

%%%%%%%%%%%%%%%%%%%%%%%
%% THIS FUNCTION LOADS AN INPUT WAV-FILE AND RETURNS A SIGNAL WHERE FLUCTUATIONS
%% IN SPECTRAL CONTENT (WITH RESPECT TO DIFFERENCES ACROSS PHASES OF THE ORIGINAL
%% SIGNAL ENVELOPE) HAVE BEEN REDUCED BY NOISE INJECTION.
%% NOTE THAT SPECTRAL CONTENT IS MADE COMPARABLE AS A FUNCTION OF PHASE (BINNED
%% INTO N BINS) OF THE SIGNAL ENVELOPE FILTERED IN A GIVEN FREQUENCY BAND f
%% REQUIRED INPUT:
%% 1. ORIGINAL INPUT SIGNAL AS WAV-FILE
%% 2. AND 3. LOWER AND UPPER FREQUENCY BOUNDARY OF f
%% 4. NUMBER OF BINS (N)
%% 5. NUMBER OF ITERATIONS USED FOR REDUCTION OF FLUCTUATIONS IN SPECTRAL CONTENT
%% THE HIGHER THIS NUMBER, THE BETTER THE PERFORMANCE OF THIS PROGRAM BUT THE 
%% SLOWER THE PROCESSING. 50 ITERATIONS USUSALLY RESULT IN AN ACCEPTABLE
%% PERFORMANCE.
%% THE RESULTING SIGNAL IS SAVED INTO THE FILE 'speechnoise.mat' AND AS WAV-FILE
%% 'speechnoise.wav'.
%%%%%%%%%%%%%%%%%%%%%%%%


%% read wav file
[signal_orig,sr] = wavread(wavfile);
signal_orig = signal_orig(:,1);
%% normalize
signal_orig = signal_orig./max(abs(signal_orig));

n_bins = n_bins+1; % first and last bins are overlapping
phases = zeros(n_bins,1);
for t = 1:n_bins
    phases(t) = -pi+(t-1)*2*pi/(n_bins-1);
end

sr_filt = sr/2;
[filt_a,filt_b]=butter(2,[freq_down/sr_filt, freq_up/sr_filt],'bandpass');
numof_freqs = 304;

%%%%%%%%%%%%%%%%%%%%%%%%%

% extract envelope and filter it in given frequency band
envelope = return_envelope(signal_orig, sr, 1);
disp('envelope extracted')
env_filt = filtfilt(filt_a,filt_b,envelope);
env_phase = angle(hilbert(env_filt));
  
%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CALCULATE TARGET SPECTRUM FOR FINAL SIGNAL

mean_spec = zeros(length(phases),numof_freqs);
counter = zeros(length(phases),1);

[WAVE,PERIOD] = contwt(signal_orig,1/sr,-1,0.05,-1,numof_freqs-1,-1,-1);   
spec_trial = abs(WAVE');
for t = 1:length(env_phase)
    % find phase bin
    [~,bin] = min(abs(env_phase(t)-phases));
    mean_spec(bin,:) = mean_spec(bin,:)+spec_trial(t,:);
    counter(bin) = counter(bin)+1;
end  

% first and last bins are overlapping
mean_spec(1,:) = mean_spec(1,:)+mean_spec(n_bins,:);
mean_spec(n_bins,:) = mean_spec(1,:);
counter(1) = counter(1)+counter(n_bins);
counter(n_bins) = counter(1);

for n = 1:n_bins
    mean_spec(n,:) = mean_spec(n,:)./counter(n);
end

target_spec = max(mean_spec);
disp('target spectrum calculated')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% CONSTRUCT FINAL SIGNAL BY ITERATIVELY REDUCING SPECTRAL CONTENT 
%% FLUCTUATIONS

% construct complementary noise
noisew = return_noise_binned(target_spec, mean_spec, sr, length(signal_orig), n_iter, n_bins, numof_freqs, env_phase); 
disp('complementary noise constructed')

% construct speech/noise mix
speechnoise = return_speech_binned(signal_orig, noisew, sr, n_iter, numof_freqs);
disp('speech/noise stimulus constructed')
%%%%%%%%%%%%%%%
 
% final correction for fluctuations
for n = 1:3
 
    [envelope2,t2] = return_envelope(speechnoise, sr, 1);

    % envelope was determined for time points t2
    % cut speech-noise mix so that its length fits with the envelope vector
    speechnoise = speechnoise(round(t2.*sr));

    % divide each point by its own energy (in the wavelet domain)
    [WAVE, PERIOD2, SCALE, COI, DJ, PARAMOUT, K] = contwt(speechnoise, 1/sr,-1,0.05,-1,numof_freqs-1,-1,-1);
    WAVE = WAVE./repmat(envelope2,[length(PERIOD2) 1]);
    speechnoise = invcwt(WAVE, 'morlet', SCALE, PARAMOUT,K);
end

%%%%%%%%%%%%%%%%%%%%%%%%
% last check: check whether spectral content is comparable across
% envelope phases

mean_spec_final = zeros(length(phases),numof_freqs);
counter_final = zeros(length(phases),1);

WAVE = contwt(speechnoise,1/sr,-1,0.05,-1,numof_freqs-1,-1,-1);   
spec_trial = abs(WAVE');
for t = 1:length(env_phase)
    % find phase bin
    [~,bin] = min(abs(env_phase(t)-phases));
    mean_spec_final(bin,:) = mean_spec_final(bin,:)+spec_trial(t,:);
    counter_final(bin) = counter_final(bin)+1;
end  

mean_spec_final(1,:) = mean_spec_final(1,:)+mean_spec_final(n_bins,:);
mean_spec_final(n_bins,:) = mean_spec_final(1,:);
counter_final(1) = counter_final(1)+counter_final(n_bins);
counter_final(n_bins) = counter_final(1);
for n = 1:n_bins
    mean_spec_final(n,:) = mean_spec_final(n,:)./counter_final(n);
end

%%%%%%%%%%%%%%%%%%%%%%%%
disp('spectra ok')
% plot spectra for all phase bins
figure
subplot(211)
semilogx(1./PERIOD,mean_spec)
xlabel('frequency')
ylabel('amplitude')
title('spectrum of the original signal for different phase bins of the original envelope')
subplot(212)
semilogx(1./PERIOD2,mean_spec_final)
xlabel('frequency')
ylabel('amplitude')
title('spectrum of the constructed signal for different phase bins of the original envelope')
%%%%%%%%%%%%%
 figure
subplot(141)
surface(phases,1./PERIOD,mean_spec')
xlabel('phase (original envelope)')
ylabel('frequency')
title('original signal')
xlim([phases(1) phases(end)])
ylim([1/PERIOD(end) 1/PERIOD(1)])
colormap('hot')
set(gca, 'YScale', 'log')
shading interp
subplot(142)
plot(mean_spec,1./PERIOD,'k')
xlabel('amplitude')
ylim([1/PERIOD(end) 1/PERIOD(1)])
set(gca, 'YScale', 'log')

 subplot(143)
surface(phases,1./PERIOD,mean_spec_final')
xlabel('phase (original envelope)')
ylabel('frequency')
title('constructed signal')
xlim([phases(1) phases(end)])
ylim([1/PERIOD(end) 1/PERIOD(1)])
 colormap('hot')
set(gca, 'YScale', 'log')
shading interp
subplot(144)
plot(mean_spec_final,1./PERIOD,'k')
xlabel('amplitude')
ylim([1/PERIOD(end) 1/PERIOD(1)])
set(gca, 'YScale', 'log')
%%%%%%%%%%%%%%%%

% normalize between -1 and 1 for sound file
speechnoise = speechnoise./max(abs(speechnoise));
save('speechnoise.mat','speechnoise*')
wavwrite(speechnoise,sr,'speechnoise.wav'); 

