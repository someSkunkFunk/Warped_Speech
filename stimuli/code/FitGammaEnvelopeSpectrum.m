clear
peak_tol = 0.1;
stimfolder = 'F:/Databases/Natural Speech/stimuli/audio/';
IPI=[];
for ii = 1:33
    audiofile = sprintf('%saudio%d.wav',stimfolder,ii);
    [audio,wav_fs]=audioread(audiofile); audio = audio(:,1);

Hd = getLPFilt(wav_fs,10);
env = abs(hilbert(audio));
env = filtfilthd(Hd,env);

env_onset = diff(env);
env_onset(env_onset<0) = 0;

[~,Ifrom,~,p] = findpeaks(env_onset,wav_fs);
p = p./std(p);

Ifrom(p<peak_tol)=[];
p(p<peak_tol)=[];

IPI = cat(1,IPI,diff(Ifrom));
end

I = 1000000;
p = gamrnd(6.3,1.05,I,1);
n = 1./p;
p0 = 1./n;
N0 = 1./gamrnd(6.3,1.05,I,1);
N1 = 1./gamrnd(3.7,2,I,1);

close all
figure
set(gcf,'Position',[600 500 1000 500])
subplot(1,2,1)
F = 0:0.005:3;
P0 = hist(N0,F);
P1 = hist(N1,F);
P = hist(IPI,F); 
plot(F,P./length(IPI),'Color',[0.8 0.8 0.8])
hold on
plot(F,P0./I)
plot(F,P1./I)
xlabel('Peak Interval (s)')
axis([0 1 0 0.045])

subplot(1,2,2)
F = 0:0.05:30;
P0 = hist(1./N0,F);
P1 = hist(1./N1,F);
P = hist(1./IPI,F); 
plot(F,P./length(IPI),'Color',[0.8 0.8 0.8])
hold on
plot(F,P0./I)
plot(F,P1./I)
xlabel('Peak Rate (Hz)')
% axis([0 1 0 0.045])