% use this scriptf - implements stretchy/compressy rules
%% TODOS: NEW IMPLEMENTATION

clear;
clc;
% userProfile=getenv('USERPROFILE');
global boxdir_lab
global boxdir_mine
inputStimfolder = sprintf('%s/stimuli/',boxdir_lab);
outputStimfolder=sprintf('%s/stimuli/wrinkle/stretchy_compressy_temp/',boxdir_mine);
%need TSM toolbox

% addpath(sprintf('%s/Box/my box/LALOR LAB/matlab-toolboxes/MATLAB_TSM-Toolbox_2.01/MATLAB_TSM-Toolbox_2.01',userProfile))
% save output to my personal box location - avoid mixing files with shared
% drive
warp_rules=[2,7];
% warp_rules=2;
% warp_rules=4; % to debug only....
% warp_rules=[1 2];
% make freqs -1, 0, or 1 to use mean median or mode
% upper/lower quartiles: 6.358/3.940 Hz (estimated using truncated gamma
% pdf fit to bark-derived peakRates)

% center_freqs=[6.358,3.940, 0];
%updated quartiles to reflect empirical distribution values
% center_freqs=[4.12960014982676,	0,	8.21688093907211];
% center_freqs=[-1 0 1] -> [lquartile median uquartile]
% -1 -> lquartile, 0->median, 1-> uquartile
center_freqs=0;
% center_freqs=[6.358,3.940, 0]; %NOTE: we assuming only 3 values given at a time MAXIMUM (one of which is median...)
% and all values greater than 1 since 1 means mode....
conditions=[1]; % 1-> irreg (stretchy) 2-> reg (compessy)
% center_freqs=logspace(0.4,.95,10);
overwrite = 1;
% stimgroup = {'leagues','oldman','wrinkle'};
stimgroup = {'wrinkle'};
% stimspeed = [0.5 2/3 3/2 2];
% seems we can bypass speed warping by feeding empty array:
stimspeed=[];

param.tolerance = 256;
param.synHop = 256;
param.win = win(1024,2); % hann window

% lowpass filter for get_peakrate:
%TODO: verify this wav_fs matches what's going into audio read
% wav_fs=44100;
% Hd = getLPFilt(wav_fs,10);

for warp_rule=warp_rules
    fprintf('warp_rule: %d\n',warp_rule)
    for ff = 1:length(center_freqs)
        center_f=center_freqs(ff);
        switch center_f
            case -1
                % center_str='mean';
                center_str='lquartile';
            case 0
                center_str='median';
            case 1
                % center_str='mode';
                center_str='uquartile';
            otherwise
                % % assuming exactly 3 different f centers in center_freqs
                % if center_f==min(center_freqs(center_freqs>1))
                %     center_str='lquartile';
                % elseif center_f==max(center_freqs)
                %     center_str='uquartile';
                % end
                % error('havent written this code')
                center_str=sprintf('%0.3fHz',center_f);
        end
        
    for gg = 1:length(stimgroup)
        fprintf('processing stimulus: %s\n',stimgroup{gg})
    
        d = dir(sprintf('%s%s/og/*.wav',inputStimfolder,stimgroup{gg}));
        for dd = 1:length(d)
            fprintf('file: %s \n',d(dd).name)
            audiofile = sprintf('%s%s/og/%s',inputStimfolder,stimgroup{gg},d(dd).name);
            [audio,wav_fs]=audioread(audiofile);
            
            % % Warp speech -> more reg
            if ismember(2,conditions)
                % reg is 2 bc we later on switched to "stretchy/compressy"
                % nomenclature which confusingly puts reg as second
                % conditioni
                type='compressy_reg';
                warp_gravity=1;
                %TODO: consolidate file dir name
                outputDirTemp=sprintf('%s%s/rule%d_seg_bark_%s/',outputStimfolder,type,warp_rule,center_str);
                audioOutputFileTemp = sprintf('%s%s',outputDirTemp,d(dd).name);
                %TODO: it's probably more efficient to store these all in a single
                %file but let's figure that out after we've debugged a bit
                audioParamsFileTemp = sprintf('%s%s',outputDirTemp,d(dd).name(1:end-4));
    
                if ~exist(audioOutputFileTemp,'file') || overwrite
                    fprintf('%s - f: %0.3f\n',type,center_f)
                    [audio_warp_temp,s_temp] = stretchyWrinkle(audio,wav_fs,warp_gravity,center_f,warp_rule); 
    
                    if ~exist(outputDirTemp,'dir')
                        mkdir(outputDirTemp)
                    end
    
                    audiowrite(audioOutputFileTemp,audio_warp_temp,wav_fs);
                    save(audioParamsFileTemp,"s_temp");
                    clear audioOutputFileTemp audioParamsFileTemp audio_warp_temp s_temp
                end
            end

            % Warp speech -> more irreg
            if ismember(1,conditions)
                type='stretchy_irreg';
                warp_gravity=-1;
                outputDirTemp=sprintf('%s%s/rule%d_seg_bark_%s/',outputStimfolder,type,warp_rule,center_str);
                
                audioOutputFileTemp = sprintf('%s%s',outputDirTemp,d(dd).name);
                %TODO: it's probably more efficient to store these all in a single
                %file but let's figure that out after we've debugged a bit
                audioParamsFileTemp = sprintf('%s%s',outputDirTemp,d(dd).name(1:end-4));
                
                if ~exist(audioOutputFileTemp,'file') || overwrite
                    fprintf('%s- f: %0.3f\n',type,center_f)
                    [audio_warp_temp,s_temp] = stretchyWrinkle(audio,wav_fs,warp_gravity,center_f,warp_rule); 
        
                    if ~exist(outputDirTemp,'dir')
                        mkdir(outputDirTemp)
                    end
        
                    audiowrite(audioOutputFileTemp,audio_warp_temp,wav_fs);
                    save(audioParamsFileTemp,"s_temp",'center_f','warp_rule');
                end
            end
    
            fprintf('\n')
        end
    end
    end
end
