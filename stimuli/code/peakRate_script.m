% script for generating peakrate variables in identical way as
% warp_stimuli_stretchy (taking after oganian et al 2019 method)
% for use in other analyses
clear
global boxdir_lab
global boxdir_mine
output_dir=fullfile(boxdir_mine,'stimuli','wrinkle','peakRate');

config_peakrate=struct('param',[],'result',[]);
config_peakrate.param.fs=128;
config_peakrate.param.experiment='fast-slow';
config_peakrate.param.peak_tol=0.1;
n_trials=75; 
% for peakrate params -- extracted matching params from original script
warp_config=get_warp_config();
switch config_peakrate.param.experiment
        case 'fast-slow'
            rate_dirs={'0.67','og','1.50'};
            rate_names={'fast','og','slow'};
            config_peakrate.result=struct();
            for rr=1:length(rate_names)
                config_peakrate.result.(rate_names{rr})=cell(1,n_trials);
            end

        otherwise
            error('yo wtf.')     
end

for rr=1:length(rate_dirs)
    rd=rate_dirs{rr};
    fprintf('getting peakrate for %s...\n',rd)
    dir_=fullfile(boxdir_lab,'stimuli','wrinkle',rd);
    wav_files=dir(dir_);
    % remove directories - assumes only files in directory
    wav_files={wav_files(~[wav_files.isdir]).name};

    for dd=1:n_trials
        wavfnm=wav_files{dd};
        fprintf('reading audiofile %d of %d...\n',dd,n_trials)
        [wf,fs]=audioread(fullfile(dir_,wavfnm));
        % calculate peakrate using exact same parameters as in
        % stretchyWrinkle (unless we've changed them since 03-05-2026)
        env=bark_env(wf,fs,config_peakrate.param.fs);
        tmp_=config_peakrate.result.(rate_names{rr});
        tmp_{dd}=get_peakRate(env, ...
            config_peakrate.param.fs,warp_config);
        config_peakrate.result.(rate_names{rr})=tmp_;
        clear env tmp_
    end
    clear dir_
end
% clear dir_ env
fprintf('saving result to %s...\n',output_dir)
save(fullfile(output_dir,'fastSlow.mat',config_peakrate));
disp('done.')

function warp_config=get_warp_config()
    warp_config=[];
    warp_config.env_lpf=14;
    warp_config.env_thresh_std=1e-3;
    warp_config.min_pkrt_height=5e-5;
end
% Oganian, Yulia, and Edward F. Chang. 
% “A Speech Envelope Landmark for Syllable Encoding in Human Superior 
% Temporal Gyrus.” SCIENCE ADVANCES, 2019.
