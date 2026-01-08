function plot_chns = normalize_channels(select_chns, n_chans)
%TODO: move this to generic utils
arguments
    select_chns {mustBeA(select_chns,{'numeric','char','string'})}
    n_chans (1,1) double {mustBePositive} = 128;
end

    if ischar(select_chns) || isstring(select_chns)
        if strcmpi(select_chns,'all')
            plot_chns = 1:n_chans;
        else
            error('Unknown channel specifier: %s', select_chns);
        end
    elseif isnumeric(select_chns)
        validateattributes(select_chns, {'numeric'}, ...
            {'integer','positive','<=',n_chans})
        plot_chns = unique(select_chns(:))'; % row vector
    else
        error('mtrf_plot_chns must be ''all'' or numeric indices');
    end
end
