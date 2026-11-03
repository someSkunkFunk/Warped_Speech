%  rectify envelopes it not already rectified
%TODO: decide if still usefull/move to stimuli or make subfunction of
%trf_analysis_script if so
clear,clc
fs=128;
envelopesFile=sprintf('../stimuli/WrinkleEnvelopes%dhz.mat',fs);
load(envelopesFile)
% should contain env, fs, and spectrogram
% only operating on env but load all so we can save them all back to the
% same file after
if any(env{1,1}<0)
    for cc=1:size(env,1)
        for ss=1:size(env,2)
            temp_env=env{cc,ss};
            temp_env(temp_env<0)=0;
            env{cc,ss}=temp_env;
        end
    end
else
    disp('it appears they are already rectified. nothing changed.')
end
clear ss cc temp_env
rectified=true;
save(envelopesFile)