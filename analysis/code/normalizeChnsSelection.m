function plotChns = normalizeChnsSelection(selectChns, nChans)
% plotChns = normalizeChnsSelection(selectChns, nChans)
arguments
    selectChns {mustBeA(selectChns,{'numeric','char','string'})}
    nChans (1,1) double {mustBePositive} = 128;
end

    if ischar(selectChns) || isstring(selectChns)
        if strcmpi(selectChns,'all')
            plotChns = 1:nChans;
        else
            error('Unknown channel specifier: %s', selectChns);
        end
    elseif isnumeric(selectChns)
        validateattributes(selectChns, {'numeric'}, ...
            {'integer','positive','<=',nChans})
        plotChns = unique(selectChns(:))'; % row vector
    else
        error('mtrf_plot_chns must be ''all'' or numeric indices');
    end
end
