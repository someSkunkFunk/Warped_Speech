function wf = addClick(wf,Fs,T,ch,click_dur)

arguments
    wf double % sound waveform. Should be 1 channel
    Fs (1,1) double % sampling rate of wf
    T (1,1) double = 0; % amount to delay wf from click onset
    ch (1,1) double = 2; % which channel is click in? 1 = what the participant hears
    click_dur = 0.01;
end

click = 0.999.*ones(click_dur*Fs,1); click(1) = 0;

if T>0 % IF we want to delay the speech AFTER the original sound waveform
    if T<click_dur % If the delay isn't long enough to account for click duration
        error('Delay, T, must be zero or longer than click duration (%0.1f). \n',click_dur)
    end
    
    % fill the click with a silent delay period
    silence_dur = T-click_dur;
    click = [click; zeros(silence_dur*Fs,1)];
    
    % Add zeros to he waveform for the delay
    wf = [zeros(size(click)); wf];
end

% Add a second channel for click if we specify
if ch==2 
    wf = [wf zeros(size(wf))];
end
wf(1:length(click),ch) = click;



