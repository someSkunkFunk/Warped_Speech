
function [envelope, t, sr_env] = return_envelope (data, sr, weighted)
% returns energy of 'data'
% input: (1) data for which the energy should be obtained, 
% (2) sampling rate of 'data', 
% (3) "1" if spectrum of 'data' should be weighted by the cochlear
% sensitivity, "0" otherwise
% output: envelope, time vector, sampling rate of the envelope 

NewFreq = [];

[WAVE,PERIOD] = contwt(data,1/sr,-1,0.05,-1,303,-1,-1);
y = abs(WAVE');
f = 1./PERIOD;
t = [1:length(data)]/sr;

ti = [];    
% compute energy for each point in time across frequency - weight by cochlea
% sensitivity, if desired
if weighted == 1
    [spl,freq] = iso226(47.747);
     spl = 47.747./spl;
     NewFreq = csapi(freq,spl,f);
     NewFreq = NewFreq / mean(NewFreq);

    ti = mean(y*NewFreq',2)';
else
    ti = mean(y,2)';
end

envelope = ti;

% sampling rate of the envelope
sr_env = 1/mean(t(2:end)-t(1:end-1));