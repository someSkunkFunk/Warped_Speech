clear
n_babble = 100;

n_stim = 120;
for ii = n_stim:-1:1
filename = sprintf('wrinkle/og/wrinkle%03d.wav',randi(120));
    [y,wav_fs] = audioread(filename);
stims(:,ii) = y;
end
len = size(stims,1);

signal = zeros(len,n_babble);

for ii = 1:n_babble
    stimI = randi(120);
    p = randi([wav_fs*2 len-wav_fs*2]);
    signal(:,ii) = stims([p:end 1:p-1],stimI);
end

signal = sum(signal,2);