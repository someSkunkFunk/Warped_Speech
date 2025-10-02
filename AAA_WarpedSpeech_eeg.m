function AAA_WarpedSpeech_eeg(subject)
%TODO: edit script to find reg/irreg stimuli
% remember to include fixed "randomized" order which avoids trials with
% non-english
try
    %% Initialize
    Screen('Preference','SyncTestSettings' ,0.002,50,0.1,5);

    datafolder = 'E:/Aaron/Warped Speech/data/';
    stimfolder = 'E:/Aaron/Warped Speech/stimuli/';
    %stimfolder = 'C:/Users/aaron/Box/Projects/Warped Speech/stimuli/';

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
    HideCursor;
    ListenChar(2);
    SetParams;

    % Flip Screen couple of times to loosen it up
    for ii = 1:10
        Screen('flip',win0);
    end

    % Initialize Sound
    chans = 1;
    InitializePsychSound;
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
    stimset = 'wrinkle'; stimfiles=[1:120]'; ntrials = 1; % 120 mins
    % set = 'oldman'; stimfiles=[1:33]'; ntrials = 6; % ~90mins
    % set = 'leagues'; stimfiles = [1:30]; % 30 mins

    % set up conditions
    m(:,1) = [2/3 1 3/2]'; % speed
    m(:,1)=[1 1 1]';
    m(:,2) = [-1 0 1]'; % regularity: 0 = normal cadence, >0 = more regular, <0 = more irregular

    if ntrials*size(m,1)>length(stimfiles)
        error('Too many trials')
    end

    % repeat for each trial
    m = repmat(m,ntrials,1);

    % shuffle
    I2 = randperm(size(m,1));
    m = m(I,:);

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
                    audiofile = sprintf('%s%s/reg/%s%03d.wav',stimfolder,stimset,stimset,tt);
                case -1
                    audiofile = sprintf('%s%s/rand/%s%03d.wav',stimfolder,stimset,stimset,tt);
            end
        else
            audiofile = sprintf('%s%s/%0.2f/%s%03d.wav',stimfolder,stimset,m(tt,1),stimset,tt);
        end

        [wf,wav_fs] = audioread(audiofile);

        % Resample if necessary
        if Fs ~= wav_fs
            wf = resample(wf,Fs,wav_fs);
        end

        % Add noise
        wf = noisySpeech(wf,Fs,SNR,speech_delay);

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
        PsychPortAudio('FillBuffer', pahandle, wf');

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
                    if x_click-text_xA<300 && x_click-text_xA>0
                        if y_click-text_yA1>0 && y_click-text_yA1<20
                            r = 1; % set response
                            c1 = cRed0; % change answer color (assumes error)
                            clicked = 1; % Trip click counter
                        elseif y_click-text_yA2>0 && y_click-text_yA2<20
                            r = 2;
                            c2 = cRed0;
                            clicked = 1;
                        elseif y_click-text_yA3>0 && y_click-text_yA3<20
                            r = 3;
                            c3 = cRed0;
                            clicked = 1;
                        elseif y_click-text_yA4>0 && y_click-text_yA4<20
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
    datafile = sprintf('%s/s%02d_WarpedSpeech.mat',datafolder,subject);

catch me
    clear wf
    datafile = sprintf('%s/s%02d_WarpedSpeechERROR.mat',datafolder,subject);
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