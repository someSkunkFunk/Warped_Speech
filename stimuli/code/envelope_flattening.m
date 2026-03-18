% envelope flattening of speech
% goald: remove slow amplitude modulations from speech signal by dividing
% by the smoothed envelope, without altering fine spectral structure

clear, close all, clc

global boxdir_mine
global boxdir_lab
n_trials=1;
%% --- Load Audio ---
wavs_dir=fullfile(boxdir_lab,'stimuli','wrinkle','og');
wavs=dir(fullfile(wavs_dir,'*.wav'));
for ii=1:n_trials
    [x, fs] = audioread(fullfile(wavs_dir,wavs(ii).name));
    % take non-click channel jk folder has stimuli without clicks
    % x = x(:,1);
end

t=(0:length(x)-1)'/fs;

disp('Loaded signal')

%% --- Extract the envelope using Hilbert Transform & smooth it---
env=abs(hilbert(x));
cutoff_hz=10;
filter_order=4;
Wn=cutoff_hz/(fs/2);
[b,a]=butter(filter_order,Wn);
env_smooth=filtfilt(b,a,env);

% avoid dividing by zero
min_envelope=0.001*max(env_smooth);

%% --- Flatten the signal by dividing ---
x_flat=x./env;


