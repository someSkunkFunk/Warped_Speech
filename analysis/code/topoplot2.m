function h = topoplot2(X,ML,EM,chanlocs,cmap)

if ~exist('ML','var')||isempty(ML)
    ML='absmax'; % Default maplimits
%     ML = [min(X) max(X)];
end

if exist('EM','var')&&isempty(EM)
    clear EM
end


% set cmap
if ~exist('cmap','var')||isempty(cmap)
    cmap = turbo(100);
    % cmap = flipud(cbrewer('seq','YlGnBu',100,'linear'));
    % cmap = flipud(cbrewer('div','RdBu',100,'linear'));
    % cmap = flipud(cbrewer('div','Spectral',100,'linear'));
end

if ~exist('chanlocs','var')||isempty(chanlocs)
    switch length(X)
        case 128
            load('128chanlocs.mat')
        case 64
            load('64chanlocs.mat')
        case 1
            schans = num2str(X);
            load([schans 'chanlocs.mat'])
            if exist('EM','var')
                X=[];
            end
        otherwise
            error('You must specify either a nchan x 1 data vector or channel n')
    end
end

set(gcf,'renderer','opengl');
if nargin==0
    error('Needs at least one input (nchans or data array)')
elseif nargin==1 && length(X)==1
    topoplot([],chanlocs,'electrodes','numbers');
else
    
    if exist('EM','var')
        if numel(EM{2})>1
        h=topoplot(X,chanlocs,'conv','off','numcontour',0,'shading','interp','verbose','off','maplimits',ML,'scatter','on','emarker',EM);
        else
        h=topoplot(X,chanlocs,'conv','off','numcontour',0,'shading','interp','verbose','off','maplimits',ML,'emarker2',EM);
        end
    else
        h=topoplot(X,chanlocs,'conv','off','numcontour',0,'shading','interp','verbose','off','maplimits',ML);
    end
colormap(cmap)
end

end
