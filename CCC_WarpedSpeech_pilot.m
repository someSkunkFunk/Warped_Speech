function CCC_WarpedSpeech_pilot(subject)

try
    test_config.shuffle=false;
    test_config.screen=false;
    test_config.time=15; %seconds
    test_config.hide_cursor=false;
    %TODO: IF adding more conditions, will need to encode them in M somehow
    test_config.reg_rule='rule5_seg_bark_median/';
    test_config.irreg_rule='rule2_seg_bark_median/';
    disp(test_config)
%% Initialize
    Screen('Preference','SyncTestSettings' ,0.002,50,0.1,5);
    datafolder = 'E:/Aaron/Warped Speech/pilot data/';
%     og_stimfolder='E:/Aaron/Warped Speech/stimuli/';
    stimfolder = 'E:/Aaron/Warped Speech/pilot stimuli/';
    % Initialize Screen
    Screen('Preference','SyncTestSettings' ,0.002,50,0.1,5);
    [hz, outRect, win0] = open_screen(2); % Aaron
    [center_x,center_y]=WindowCenter(win0);
    if test_config.hide_cursor
        HideCursor;
    end
    ListenChar(2);
    SetParams; %Aarn
    % Flip Screen couple of times to loosen it up
    for ii = 1:10
        Screen('flip',win0);
    end
    % Initialize Sound
    chans = 1; %NOTE: does not work when using single channel audio file
    InitializePsychSound;
    pahandle = PsychPortAudio('Open', deviceID, 1, 2,Fs, chans);
    
    % define number of samples to grab
    n_samples=round(Fs*test_config.time);

    % Play silence to loosen it up
    silence = zeros(chans,50);
    PsychPortAudio('FillBuffer', pahandle, silence);
    PsychPortAudio('Start', pahandle, 1);
    WaitSecs(0.5);
    % Set up trial structure.
    SNR = -3;
    speech_delay = 1;
    % parameters depend on dataset used
    stimset = 'wrinkle'; stimfiles=[1:120]'; ntrials = 3; % 120 mins
    m(:,1) = [1 1 1];
    m(:,2) = [-1 0 1]'; % regularity: 0 = normal cadence, >0 = more regular, <0 = more irregular
    if ntrials*size(m,1)>length(stimfiles)
        error('Too many trials')
    end
    % repeat for each trial
    m = repmat(m,ntrials,1);
    % shuffle
    if test_config.shuffle
        I = randperm(size(m,1));
        m = m(I,:);
    end
    totaltrials = size(m,1);
    % Wait for participant input
    Screen('DrawText', win0, 'Press any key to begin the experiment.',400,500,cWhite0);
    Screen('Flip',win0);
    keytest_unbound; % SEND TO EMILY
    Screen('Flip',win0);
    %% Experiment trials
    for tt = 1:totaltrials
        %% Pre-stimulus setup
        fprintf('%s, tr = %d',datestr(now,13), tt)
        % Acquire stimuli
        if m(tt,1)==1
            switch m(tt,2)
                case 0
                    audiofile = sprintf('%s%s/og/%s%03d.wav','./stimuli/',stimset,stimset,tt);
                case 1
                    audiofile = sprintf('%s%s/reg/%s%s%03d.wav',stimfolder,stimset, ...
                        test_config.reg_rule,stimset,tt);
                case -1
                    audiofile = sprintf('%s%s/irreg/%s%s%03d.wav',stimfolder,stimset, ...
                        test_config.irreg_rule,stimset,tt);
            end
        else
            error('nah bro')
%             audiofile = sprintf('%s%s/%0.2f/%s%03d.wav',stimfolder,stimset,m(tt,1),stimset,tt);
        end
        [wf,wav_fs] = audioread(audiofile,[1 n_samples]);
        % Resample if necessary
        if Fs ~= wav_fs 
            wf = resample(wf,Fs,wav_fs);
        end
    
    wav_nchans=size(wf,2);
    switch wav_nchans
        case 1
            % no click, just add noise
            wf=noisySpeech(wf,Fs,SNR,speech_delay);
        case 2
            % take click out and put back in after adding noise to avoid chaos
            % assuming click is in channel 2
            wf_click=wf(:,2);
            wf=noisySpeech(wf(:,1),Fs,SNR,speech_delay);
            
            % click in channel 2 (right)
            wf=[wf,[zeros(length(wf)-length(wf_click),1);wf_click;]];
            % click in channel 1 (left)
            % wf=[[zeros(length(wf)-length(wf_click),1);wf_click;],wf];
    end

    % Put up fixation point
    Screen('DrawDots', win0,[0 0] ,fixsize ,fixcolor ,scrcenter ,1);
    Screen('Flip',win0);
    % Jittered fixation - NOTE: what this do?
    WaitSecs(2);
    % Take control over processor - NOTE: do we need?
    Priority(2);
    % Fill auditory buffer
    PsychPortAudio('FillBuffer', pahandle, wf');
    %% Stimulus
    % Play auditory
    PsychPortAudio('Start', pahandle, 1, GetSecs+0.5); % Add time here to delay auditory
    WaitSecs(0.5);
    WaitSecs(0.5);
    % NOTE: stuff below needed?
    status = PsychPortAudio('GetStatus', pahandle);
    esc = 0;
    while status.Active
        status = PsychPortAudio('GetStatus', pahandle);
        r = keytest_unbound(0.5);
        % Escape block
        if strcmp('ESCAPE',r)
            esc = 1;
            break
        end
        % skip block
        if strcmp('s',r)
            break
        end
    end
    %% Post-stimulus junk
    % Stop audios
    PsychPortAudio('Stop',pahandle);
    % Release control of processor
    Priority(0);
    % Post-stimulus period
    WaitSecs(2);
    % Execute escape
    if esc
        break
    end
    end


    %% finish and close experiment
    % clear large useless variables
    clear wf
    datafile=sprintf('%s/s%02d_pilot.mat/',datafoder,subject);
catch me
    clear wf
    datafile = sprintf('%s/s%02d_pilotERROR.mat',datafolder,subject);
end

% save workspace to disk
save(datafile)

% Close routine
ShowCursor;
ListenChar(0);
Screen('CloseAll')
PsychPortAudio('Close')

if exist('me','var')
    rethrow(me)
end
end