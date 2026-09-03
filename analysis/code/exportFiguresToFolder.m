function exportFiguresToFolder(targetFolder, varargin)
% EXPORTFIGURESTOFOLDER Exports all open figures to a specific folder,
% optionally saves a metadata struct, and CLOSES all figures to free RAM
%
% Usage:
%   export FiguresToFolder(targetFolder)
%   exporrtFiguresToFolder(targetFolder,metadataStruct)

% ensure target directory exists
if ~exist(targetFolder, 'dir')
    mkdir(targetFolder);
end

% find all open figures
allFigs=findobj('Type','figure');
nFigs=numel(allFigs);

if nFigs==0
    fprintf('No open figures to export to %s.\n',targetFolder)
    return;
end


fprintf('Exporting %d figures to %s...\n',nFigs,targetFolder)

% loop thru figs and export

for ii=1:nFigs
    fig=allFigs(ii);
    figName=fig.Name;
    % clean the figure name to make it a valid, readablefilename
    cleanName=strrep(figName,' | ','_');
    cleanName=strrep(cleanName,' ','_');
    % note: not used in this script but probably good to generally avoid OS
    % issues with unacceptable filenames
    cleanName=strrep(cleanName,':','');
    cleanName=strrep(cleanName,'/','_');
    
    % export as high-res PNG
    exportgraphics(fig, fullfile(targetFolder, [cleanName '.png']), ...
        'Resolution', 300);


end
% save metadata if provided
if ~isempty(varargin)
    metadataStruct=varargin{1};
    metaFile=fullfile(targetFolder, 'metadata.mat');
    save(metaFile,'metadataStruct');
    fprintf('Saved metadata to %s\n', metaFile);
end
% close all figures to free up RAM
close(allFigs);
fprintf('Closed %d figures to free memory.\n',nFigs);
end