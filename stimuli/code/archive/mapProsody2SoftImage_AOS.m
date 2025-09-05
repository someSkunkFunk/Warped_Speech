function [softImageLabels,freq,phMap] = mapProsody2SoftImage_AOS(prosodyLabels)
    softImageLabels = zeros(1,length(prosodyLabels));

    tableMap{1} = {'sp';'aa';'ae';'ah';'ao';'aw';'ay';'b';'ch';'d';'dh';'eh';'er';'ey';'f';'g';'hh';'ih';'iy';...
        'jh';'k';'l';'m';'n';'ng';'ow';'oy';'p';'r';'s';'sh';'t';'th';'uh';'uw';'v';'w';'y';'z';'zh'}; 
    %39 phonemes in tableMap - sp is not a real phoneme
    phMap = 0:39;

    prosodyLabels = upper(prosodyLabels); %converts any lower case strings to upper case
    %prosodyLabels2 = cellfun(@(s) s(1:end-1*isnumeric(s(end))), prosodyLabels);
    for i = 1:length(prosodyLabels) % removing numbers from phoneprosodyLabelsme labels
        if ~isempty(prosodyLabels{i}) && (prosodyLabels{i}(end) == '0' || prosodyLabels{i}(end) == '1' || prosodyLabels{i}(end) == '2' || prosodyLabels{i}(end) == '3')
            prosodyLabels{i} = prosodyLabels{i}(1:end-1); %removing last string of label, if its a number
        end
    end

    tableMap{1} = upper(tableMap{1});
    for i = 1:length(tableMap{1})
        %this finds the indices of the tableMap phonemes in prosodyLabels -
        %which are the labels taken from fave align
        ispresent = cellfun(@(s) strcmp(tableMap{1}{i}, s), prosodyLabels);
        %then the softImageLabels contains the phIndex for that phoneme
        softImageLabels(ispresent) = phMap(i);
        freq(i) = sum(ispresent); % how many of this element
    end
end













