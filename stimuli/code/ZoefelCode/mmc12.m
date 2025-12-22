function [speech,WAVE] = return_speech_binned (data, noise, sr, numof_it, numof_freqs)

% mix in the wavelet domain

[WAVE1,PERIOD1,SCALE1,COI1,DJ1, PARAMOUT1, K1] = contwt(data,1/sr,-1,0.05,-1,numof_freqs-1,-1,-1);

WAVE2 = noise;

factor = abs(WAVE1) + abs(WAVE2);

WAVE = WAVE1 + WAVE2;

for g = 1:numof_it
     if g == round(numof_it/4)
        disp('speech/noise construction: 25%')
    elseif g == round(numof_it/2)
        disp('speech/noise construction: 50%')
    elseif g == round(3*numof_it/4)
        disp('speech/noise construction: 75%')
    end
    WAVE = (WAVE ./ abs(WAVE)) .* factor;
    speech = invcwt(WAVE, 'morlet', SCALE1, PARAMOUT1,K1);
    [WAVE,PERIOD1,SCALE1,COI1,DJ1, PARAMOUT1, K1] = contwt(speech,1/sr,-1,0.05,-1,numof_freqs-1,-1,-1);
end

end