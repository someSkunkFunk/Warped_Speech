% 64s is known duration of the original stimuli
%% get durations of wavs in folder:
clear, close all
%%
% select which audio set to look at
global boxdir_mine
fldr=[boxdir_mine '\stimuli\wrinkle\stretchy_compressy_temp\stretchy_irreg\rule2_seg_bark_median_unnormalized_durations'];
% fldr=[boxdir_mine '\stimuli\wrinkle\stretchy_compressy_temp\stretchy_irreg\rule7_seg_bark_median_unnormalized_durations'];
D=dir([fldr, '\*.wav']);
durations=zeros(size(D));
for dd=1:numel(D)
    wav_path=[fldr,'\',D(dd).name];
    [wav,fs]=audioread(wav_path);
    durations(dd)=(length(wav)-1)/fs;
end
%%
figure
scatter(1:length(durations),durations)

fprintf('avg dur: %0.3f\n',mean(durations))


interval_factor=durations./64;
corrected_durations=durations./interval_factor;

output_fldr=[boxdir_mine '\stimuli'];