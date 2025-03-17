function normalizedShit=normalizeByArea(unnormalizedYvals,xVals)
% TODO: check that peakRateINtervalsGammaFitBroadband.m still works with
% maybe move to dependencies - or some location such that we can access
% regardless of project...
% this function being outside the script now
% function normalizedShit=normalizeByArea(unnormalizedYvals,xVals)
% unnormalizedShit: y values
% xvals: need to all be uniformly spaced (i think)
    xDeltas=round(diff(xVals),5); 
    if ~all(xDeltas==xDeltas(1))
        error('dude')
    else
        xDelta=xDeltas(1);
    end

    normalizedShit=unnormalizedYvals./(xDelta.*trapz(unnormalizedYvals));
end