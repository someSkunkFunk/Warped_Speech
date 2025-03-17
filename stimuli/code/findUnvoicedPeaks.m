clear; clc;

peak_tol = 0.1;
env_fs = 2048;
stimfolder = 'F:/Databases/Natural Speech/stimuli/';

file = 1;

audiofile = sprintf('%saudio/audio%d.wav',stimfolder,file);
feafile = sprintf('%sPhonetic Features/oldMan_128Hz_%d_fea.mat',stimfolder,file);
gridfile = sprintf('%stextGrid/audio%d.textGrid',stimfolder,file);

[audio,wav_fs]=audioread(audiofile); audio = audio(:,1);

[labels, start, stop, name] = readTextgrid(gridfile);

ph_label = labels{1}; ph_time = start{1};
Isil = ismember(ph_label,'sil');
Iempty = ismember(ph_label,'');

ph_label(Isil|Iempty) = [];
ph_time(Isil|Iempty) = [];

for ii = 1:length(ph_label)
    ph_label{ii} = ph_label{ii}(~ismember(ph_label{ii},'1234567890'));
end

Hd = getLPFilt(wav_fs,10);
env = abs(hilbert(audio));
env = filtfilthd(Hd,env);
env_onset = diff(env);
env_onset(env_onset<0) = 0;
[~,Ifrom,~,p] = findpeaks(env_onset,wav_fs);

% normalize prominence
p = p./std(p);

% Eliminate small peaks
Ifrom(p<peak_tol)=[];
p(p<peak_tol)=[];

load(feafile)
temp = diff(feaStim,[],2);
temp(temp==-1)= 0;
ph_onset = [0 sum(temp)]; clear temp

Iv = 7;
Iuv = 8;

voicedTime = (find(diff(feaStim(Iv,:))>0)+1)./128;
unvoicedTime = (find(diff(feaStim(Iuv,:))>0)+1)./128;
phTime = find(ph_onset)./128;


env = resample(env,env_fs,wav_fs);
env_onset = resample(env_onset,env_fs,wav_fs);

t = linspace(0,length(audio)./wav_fs,length(audio));
t_env = linspace(0,length(env)./env_fs,length(env));

figure
hold on
plot(t_env,env./std(env),'Color',[210 220 240]./255)
plot(t_env(1:end-1),env_onset./std(env_onset),'Color',[250 200 190]./255)
stem(Ifrom,8.*ones(size(Ifrom)),'.','Color',[80 80 80]./255,'LineWidth',2)

stem(phTime,7.*ones(size(phTime)),'.','Color',[150, 150, 150]./255,'LineWidth',2)
stem(voicedTime,7.*ones(size(voicedTime)),'.','Color',[150, 0, 200]./255,'LineWidth',2)
stem(unvoicedTime,7.*ones(size(unvoicedTime)),'.','Color',[210, 80, 0]./255,'LineWidth',2)
text(ph_time,7.5.*ones(size(ph_time)),ph_label)