% EXAMPLE AM-MODULATION SCRIPT FROM AARON
dur = 5; % seconds
fs = 44100;


f = [10 30]; % limits of frequencies in AM signal

% frequency and time domain continua
F = linspace(0,fs,fs*dur);
T = linspace(1./fs,dur,dur*fs);

% define the spectrum (maybe this can be non-discrete?)
fspec = zeros(1,fs*dur);
fspec(F>=f(1)&F<=f(2)) = 1;

% define random phase for each freq bin
p = (rand(size(fspec))*2-1)*pi;
a = sqrt(pi.^2-p.^2).* sign(randn(size(fspec)));
z = complex(a,p);

% inverse to time domain with only the selected freqs
envelope = ifft(fspec.*z,'symmetric');

% Generate noise/sinewave carrier signal
carrier = randn(size(envelope));

% Generate onset/offset gate
gatetime = 0.01;
gatesamp = round(gatetime.*fs);
rampu = linspace(1./fs,1,gatesamp);
rampd = linspace(1,1./fs,gatesamp);
gate = [rampu ones(1,(fs*dur-2.*gatesamp)) rampd];

% Apply envelopes and gate to carrier signal
AMnoise = (1+(envelope./-min(envelope))).*carrier.*gate;

figure('Name', 'AMNoise waveform')
ax1=subplot(2,1,1);
plot(F, fspec, 'Color', 'red')
title('|AM F|')
xlabel('frequency (hz)')
xlim([0, 40])
ax2=subplot(2,1,2);
plot(T,normalize(envelope,'range'),'r--'); hold on
plot(T,rescale(AMnoise,-1,1),'k')
legend('AM envelope', 'AMNoise')
xlabel('time (s)')
title('envelope')
soundsc(AMnoise,fs)
