function BBB_WarpedSpeech_pilot(subject)
% for pilot testing rhythmic vs arhythmic stimuli
% todo: used fixed permutation of conditions so every subject has the same
% once it works, replace AAA script with this

try
    config.shuffle=true;
    % script_config.screen=false;
    config.experiment='reg-irreg';
    config.skip_questions=false;
    config.ntrials_per=25;
    % script_config.trials_start=50;
    config.add_noise=false;
    config.hide_cursor=true   ;
    config.short_trial_dur=[]; % in seconds; empty if full trial dur
    config.show_clickboxes=false;
    disp(config)
    %% Initialize
    Screen('Preference','SyncTestSettings' ,0.002,50,0.1,5);

    datafolder = 'E:/Aaron/Warped Speech/data/';
    stimfolder = 'E:/Aaron/Warped Speech/stimuli_wClicks/';
    
    % Initialize the serial port interface (COM4 is connected with the USB receiver)
    srlprt=serial(biosemiComPort,'BaudRate',115200,'DataBits',8,'StopBits',1,'Parity','none');
    fopen(srlprt);

    trghi = 255;
    trglo = 0;
    pauseoff = 126; % trigger for recording start
    pauseon = 127; % trigger for recording stop
    resptrg = 128; % trigger for subject response

    % Trigger a few times to loosen it up
    for ii = 1:10
        fwrite(srlprt,trghi)
        WaitSecs(0.02);
    end

    % Initialize Screen
    Screen('Preference','SyncTestSettings' ,0.002,50,0.1,5);
    [hz, outRect, win0] = open_screen(2);
    [center_x,center_y]=WindowCenter(win0);
    if config.hide_cursor
        HideCursor;
    end

    ListenChar(2);
    
    % Flip Screen couple of times to loosen it up
    for ii = 1:10
        Screen('flip',win0);
    end

    % Initialize Sound
    InitializePsychSound;
    try
        SetParams;
    catch ME
        fprintf('SetParams error: %s\n',ME.message)
    end
    chans = 2;
    pahandle = PsychPortAudio('Open', deviceID, 1, 2,Fs, chans);

    % Play silence to loosen it up
    silence = zeros(chans,50);
    PsychPortAudio('FillBuffer', pahandle, silence);
    PsychPortAudio('Start', pahandle, 1);
    WaitSecs(0.5);

    % Set up trial structure.
    % clear
    SNR = -3;
    speech_delay = 1;

    % parameters depend on dataset used
    stimset = 'wrinkle'; stimfiles=[1:120]'; ntrials = config.ntrials_per; % 120 mins
    % set = 'oldman'; stimfiles=[1:33]'; ntrials = 6; % ~90mins
    % set = 'leagues'; stimfiles = [1:30]; % 30 mins

    % set up conditions
    switch lower(config.experiment)
        % m(:,1): time-compression
         % m(:,2): regularity: 0 = normal cadence, <0 = more regular, >0 = more irregular
        case 'fast-slow'
            m(:,1) = [2/3 1 3/2]'; 
            m(:,2) = [0 0 0]';
        case 'reg-irreg'
            m(:,1) = [1 1 1]';
            m(:,2) = [-1 0 1]';
    end

    if ntrials*size(m,1)>length(stimfiles)
        error('Too many trials')
    end

    % repeat for each trial
    m = repmat(m,ntrials,1);

    % shuffle
    if config.shuffle
        I = randperm(size(m,1));
        m = m(I,:);
    end
    totaltrials = size(m,1);

    % Grab comprehension questions:
    T = readtable(sprintf('%s%s/questions.xlsx',stimfolder,stimset));

    % Display subject ID
    id = generateIDs(subject);
    fprintf('***************************\n')
    fprintf('*    Subject ID = %s     *\n',id)
    fprintf('***************************\n')

    % Provide instructions
    instructions;

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
                    audiofile = sprintf('%s%s/og/%s%03d.wav',stimfolder,stimset,stimset,tt);
                case 1
                    audiofile = sprintf('%s%s/rand/%s%03d.wav',stimfolder,stimset,stimset,tt);
                case -1
                    audiofile = sprintf('%s%s/reg/%s%03d.wav',stimfolder,stimset,stimset,tt);
            end
        else
            audiofile = sprintf('%s%s/%0.2f/%s%03d.wav',stimfolder,stimset,m(tt,1),stimset,tt);
        end

        [wf,wav_fs] = audioread(audiofile);

        % Resample if necessary
        if Fs ~= wav_fs
            wf = resample(wf,Fs,wav_fs);
        end

        wav_nchans=size(wf,2);
        if config.add_noise
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
        end

        % Put up fixation point
        Screen('DrawDots', win0,[0 0] ,fixsize ,fixcolor ,scrcenter ,1);
        Screen('Flip',win0);

        % Trigger recording start
        fwrite(srlprt,pauseoff);

        % Jittered fixation
        WaitSecs(2);

        % Take control over processor
        Priority(2);

        % Stimulus Code
        stimtrg = tt;

        % Fill auditory buffer
        if isempty(config.short_trial_dur)
            % play full recording           
            PsychPortAudio('FillBuffer', pahandle, wf');
        else
            nclip_=config.short_trial_dur*Fs;
            PsychPortAudio('FillBuffer', pahandle, wf(1:nclip_,:)');
            clear nclip_
        end

        %% Stimulus
        % Play auditory
        PsychPortAudio('Start', pahandle, 1, GetSecs+0.5); % Add time here to delay auditory
        WaitSecs(0.5);
        % EEG trigger for stimulus start (2.35 ms)
        fwrite(srlprt,stimtrg);

        WaitSecs(0.5);
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

        % Trigger recording pause
        fwrite(srlprt,pauseon);

        % Execute escape
        if esc
            break
        end

        %% Comprehension questions
        if ~config.skip_questions
            nQuestions = 2; % max 5
    
            if ~mod(tt,2) % Only every other trial
    
                qIs = find(T.trial==tt); I = datasample(qIs,nQuestions,'Replace',false);
    
                ShowCursor('CrossHair');
                SetMouse(center_x,center_y,win0);
                temp = zeros(1,4); % for storing 2 qa answers & responses
    
                for qq = 1:nQuestions
    
                    question = T.question{I(qq)};
                    answers{1} = T.A1{I(qq)};
                    answers{2} = T.A2{I(qq)};
                    answers{3} = T.A3{I(qq)};
                    answers{4} = T.A4{I(qq)};
                    correct_ans = T.ans(I(qq));
    
                    % Set answer color (white to start)
                    cQ = cWhite0;
                    c1 = cWhite0;
                    c2 = cWhite0;
                    c3 = cWhite0;
                    c4 = cWhite0;
    
                    % Determine order of answers,
                    Aorder = randperm(4);
    
                    % Extract correct answer position
                    temp(qq) = find(Aorder==correct_ans);
    
                    % Display question and 4 answers
                    Screen('DrawText', win0,question,text_xQ,text_yQ,cQ);
                    Screen('DrawText', win0,answers{Aorder(1)},text_xA,text_yA1,c1);
                    Screen('DrawText', win0,answers{Aorder(2)},text_xA,text_yA2,c2);
                    Screen('DrawText', win0,answers{Aorder(3)},text_xA,text_yA3,c3);
                    Screen('DrawText', win0,answers{Aorder(4)},text_xA,text_yA4,c4);
                    clickbox_width=300;
                    clickbox_height=40;
                    % include boxes to visualize clickable area
                    if config.show_clickboxes
                        try
                            Screen('FrameRect', win0,[0 255 0],[text_xA, text_yA1, ...
                                text_xA+clickbox_width, text_yA1+clickbox_height]);
                            Screen('FrameRect', win0,[0 255 0],[text_xA, text_yA2, ...
                                text_xA+clickbox_width, text_yA2+clickbox_height]);
                            Screen('FrameRect', win0,[0 255 0],[text_xA, text_yA3, ...
                                text_xA+clickbox_width, text_yA3+clickbox_height]);
                            Screen('FrameRect', win0,[0 255 0],[text_xA, text_yA4, ...
                                text_xA+clickbox_width, text_yA4+clickbox_height]);
                            
                        catch
                            disp('rect values given')
                            disp([text_xA, clickbox_height+text_yA1, ...
                                text_xA+clickbox_width,text_yA1])
                        end
                    end
                    Screen('Flip', win0);
                    % Set all answer colors to background (disappear) later set correct ans (green) and error (red)
                    c1 = cBg0;
                    c2 = cBg0;
                    c3 = cBg0;
                    c4 = cBg0;
    
                    % reset click counter
                    clicked = 0;
    
                    % collect clicks until it overlaps with an answer

                    while ~clicked
                        [~,x_click,y_click] = GetClicks;
                        % if click is to the right of text_xA by
                        % click_width or less pixels
                        if x_click-text_xA<clickbox_width && x_click-text_xA>0
                            if y_click-text_yA1>0 && y_click-text_yA1<clickbox_height
                                r = 1; % set response
                                c1 = cRed0; % change answer color (assumes error)
                                clicked = 1; % Trip click counter
                            elseif y_click-text_yA2>0 && y_click-text_yA2<clickbox_height
                                r = 2;
                                c2 = cRed0;
                                clicked = 1;
                            elseif y_click-text_yA3>0 && y_click-text_yA3<clickbox_height
                                r = 3;
                                c3 = cRed0;
                                clicked = 1;
                            elseif y_click-text_yA4>0 && y_click-text_yA4<clickbox_height
                                r = 4;
                                c4 = cRed0;
                                clicked = 1;
                            end
                        end
                    end
    
                    % Record response
                    temp(qq+nQuestions) = r;
    
                    % Set correct answer color (overwrites click color if no error)
                    switch temp(qq)
                        case 1
                            c1 = cGreen0;
                        case 2
                            c2 = cGreen0;
                        case 3
                            c3 = cGreen0;
                        case 4
                            c4 = cGreen0;
                    end
    
                    % Display feedback
                    WaitSecs(0.1);
                    Screen('DrawText', win0,question,text_xQ,text_yQ,cQ);
                    Screen('DrawText', win0,answers{Aorder(1)},text_xA,text_yA1,c1);
                    Screen('DrawText', win0,answers{Aorder(2)},text_xA,text_yA2,c2);
                    Screen('DrawText', win0,answers{Aorder(3)},text_xA,text_yA3,c3);
                    Screen('DrawText', win0,answers{Aorder(4)},text_xA,text_yA4,c4);
                    Screen('Flip', win0);
    
                    WaitSecs(2);
                end
    
                HideCursor;
    
                % transfer temp to m
                I = 3:(3+2*nQuestions-1);
                m(tt,I) = temp;
    
                % Report QA results to experimenter
                nCorrect = sum(temp(1:nQuestions)==temp(nQuestions+(1:nQuestions)));
                fprintf(', QA = %d/%d.',nCorrect,nQuestions);
            end
        end
        fprintf('\n')

        %% Break
        % Give break every block (ntrials)
        if tt>1 && mod(tt,totaltrials)==0
            Screen('DrawText', win0, 'You''re done! Thanks for participating!.',400,400,cWhite0);
            Screen('DrawText', win0, 'Press any key to exit the experiment.',400,500,cWhite0);
            Screen('Flip',win0);
            keytest_unbound;

        else
            Screen('DrawText', win0, ['Finished with trial ', num2str(tt),' of ',num2str(totaltrials),'.'],400,400,cWhite0);
            Screen('DrawText', win0, 'Press any key to start the next trial.',400,500,cWhite0);
            Screen('Flip',win0);
            keytest_unbound;

        end
        Screen('Flip',win0);
    end

    %% Finish up and close experiment
    
    % clear large useless variables
    clear wf
    % define save filename
    datafile = sprintf('%s/s%02d_RegIrregPilot.mat',datafolder,subject);

catch me
    clear wf
    datafile = sprintf('%s/s%02d_RegIrregPilotERROR.mat',datafolder,subject);
end

% save workspace to disk
save(datafile)

% Close routine
ShowCursor;
ListenChar(0);
fclose(srlprt);
Screen('CloseAll')
PsychPortAudio('Close')

if exist('me','var')
    rethrow(me)
end