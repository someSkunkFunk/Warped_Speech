% script for generating peakrate variables in identical way as
% warp_stimuli_stretchy (taking after oganian et al 2019 method)
% for use in other analyses
global boxdir_lab


config_peakrate=struct('param',[],'result',[]);
config_peakrate.param.experiment='fast-slow';
n_trials=75; 

switch config_peakrate.param.experiment
        case 'fast-slow'
            rate_dirs={'0.67','og','1.50'};
            rate_names={'fast','og','slow'};
            config_peakrate.result=cell2struct(cell(3,n_trials),rate_names);

        otherwise
            error('yo wtf.')     
end

for rr=1:length(rate_dirs)
    rd=rate_dirs{rr};
    fprintf('getting peakrate for %s...\n',rd)
    wav_files=dir(fullfile(boxdir_lab,'stimuli','wrinkle',rd));
    % remove directories - assumes only files in directory
    wav_files=wav_files(~[wav_files.isdir]).name;

    for dd=1:length(wav_files)
        wavfnm=wav_files(dd);
        fprintf('reading audiofile %d of %d...\n',dd,length(wav_files))
        [wf,fs]=audioread(wavfnm);
        % calculate peakrate using exact same parameters as in
        % stretchyWrinkle (unless we've changed them since 03-05-2026)
        env=bark_env(wf,fs,fs);
        [peakrate_times,peakrate_amps]=get_peakRate(env,fs,peak_tol);
    end
end


% Oganian, Yulia, and Edward F. Chang. 
% “A Speech Envelope Landmark for Syllable Encoding in Human Superior 
% Temporal Gyrus.” SCIENCE ADVANCES, 2019.
