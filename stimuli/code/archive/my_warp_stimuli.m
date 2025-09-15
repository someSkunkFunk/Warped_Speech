% % andre's copy of warp stimuli to avoid making changes we want to undo and
% % NOT UPDATED - DO NOT USE
% %% TODOS: NEW IMPLEMENTATION
% % 1. fixed percent shift to or away from mean/median/mode based on if above below mean/median/mode
% % 
% % 2. precent shift depending on distance from mean/median/mode on peak intervals
% % 
% % 3. percent or fixed shift to or away from mean/median/mode but make it somewhat random and not deterministic
% % 
% % all of these could be applied on the interval distribution directly or inverted from frequency distribution - should try both and compare/contrast 
% clear;
% clc;
% userProfile=getenv('USERPROFILE');
% inputStimfolder = sprintf('%s/Box/Lalor Lab Box/Research Projects/Aaron - Warped Speech/stimuli/',userProfile);
% 
% %need TSM toolbox
% addpath(sprintf('%s/Box/my box/LALOR LAB/matlab-toolboxes/MATLAB_TSM-Toolbox_2.01/MATLAB_TSM-Toolbox_2.01',userProfile))
% % save output to my personal box location - avoid mixing files with shared
% % drive
% outputStimfolder=sprintf('./rewarped/');
% 
% overwrite = 1;
% % stimgroup = {'leagues','oldman','wrinkle'};
% stimgroup = {'wrinkle'};
% % stimspeed = [0.5 2/3 3/2 2];
% % seems we can bypass speed warping by feeding empty array:
% stimspeed=[];
% 
% param.tolerance = 256;
% param.synHop = 256;
% param.win = win(1024,2); % hann window
% for gg = 1:length(stimgroup)
%     fprintf('processing stimulus: %s\n',stimgroup{gg})
% 
%     d = dir(sprintf('%s%s/og/*.wav',inputStimfolder,stimgroup{gg}));
%     for dd = 1:length(d)
%         fprintf('file: %s ',d(dd).name)
%         audiofile = sprintf('%s%s/og/%s',inputStimfolder,stimgroup{gg},d(dd).name);
%         [audio,wav_fs]=audioread(audiofile);
% 
%         % Warp Speech speed, natural cadence
%         for ss = 1:length(stimspeed)
%             savefile = sprintf('%s%s/%0.2f/%s',outputStimfolder,stimgroup{gg},stimspeed(ss),d(dd).name);
%             if ~exist(savefile,'file') || overwrite
% 
%                 fprintf('%0.2fx ',stimspeed(ss))
%                 audio_warp = wsolaTSM(audio,stimspeed(ss),param);
% 
%                 if ~exist(sprintf('%s%s/%0.2f/',outputStimfolder,stimgroup{gg},stimspeed(ss)),'dir')
%                     mkdir(sprintf('%s%s/',outputStimfolder,stimgroup{gg}),sprintf('%0.2f',stimspeed(ss)))
%                 end
% 
%                 audiowrite(savefile,audio_warp,wav_fs);
%             end
%         end
% 
%         % Natural Speed, Warp Candence
% 
%         % % Make more regular NOTE: muted because uncertain how to handle
%         % mean case here still
%         % type = 'reg';
%         % savefile = sprintf('%s%s/%s/%s',outputStimfolder,stimgroup{gg},type,d(dd).name);
%         % 
%         % if ~exist(savefile,'file') || overwrite
%         %     fprintf('%s ',type)
%         %     audio_warp = rhythmicWrinkle(audio,wav_fs,1); 
%         % 
%         %     if ~exist(sprintf('%s%s/%s/',outputStimfolder,stimgroup{gg},type),'dir')
%         %         mkdir(sprintf('%s%s/',outputStimfolder,stimgroup{gg}),type)
%         %     end
%         % 
%         %     audiowrite(savefile,audio_warp,wav_fs);
%         % end
% 
%         % Shuffle intervals randomly using original distribution
%         type = 'randSame2';
%         rand_how=1;
%         savefile = sprintf('%s%s/%s/%s',outputStimfolder,stimgroup{gg},type,d(dd).name);
% 
%         if ~exist(savefile,'file') || overwrite
%             fprintf('%s ',type)
%             audio_warp = rhythmicWrinkle(audio,wav_fs,-1,rand_how); 
% 
%             if ~exist(sprintf('%s%s/%s/',outputStimfolder,stimgroup{gg},type),'dir')
%                 mkdir(sprintf('%s%s/',outputStimfolder,stimgroup{gg}),type)
%             end
% 
%             audiowrite(savefile,audio_warp,wav_fs);
%         end
%         % Shuffle randomly using uniform distribution
%         type = 'randUnif';
%         savefile = sprintf('%s%s/%s/%s',outputStimfolder,stimgroup{gg},type,d(dd).name);
%         rand_how=2;
% 
%         if ~exist(savefile,'file') || overwrite
%             fprintf('%s ',type)
%             audio_warp = rhythmicWrinkle(audio,wav_fs,-1,rand_how); 
% 
%             if ~exist(sprintf('%s%s/%s/',outputStimfolder,stimgroup{gg},type),'dir')
%                 mkdir(sprintf('%s%s/',outputStimfolder,stimgroup{gg}),type)
%             end
% 
%             audiowrite(savefile,audio_warp,wav_fs);
%         end
% 
%         fprintf('\n')
%     end
% end
% 
