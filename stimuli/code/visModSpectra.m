% visModSpectra
%DONE: add ability to plot:
% 1. MS of same condition as function of rule
% 2. MS of same condition as function of frequency center
%REMAINING TODO:
% add a dummy check to ensure we don't feed multiple conditions (or single
% condition) with multiple rules or frequencies
clear, clc
ms_dir='./modSpectra';
warped_ms_flpth=sprintf('%s/allStretchyCompressyMS.mat',ms_dir);
og_ms_flpth=sprintf('%s/ogWrinkleDingMS.mat',ms_dir);
show_fcenters={'median','lquartile','uquartile'}; % median, lquartile, or uquartile
show_rules=2;
show_conditions={'irregular'}; 
n_lines=numel(show_conditions)*numel(show_rules)*numel(show_fcenters);
warped_ms=load(warped_ms_flpth);
warped_ms=warped_ms.modSpectra;
og_ms=load(og_ms_flpth);
freqs=og_ms.freqs;
og_ms=og_ms.modSpectra.og;

normByArea=false;
include_og=true;
logTicks=2.^(-2:5);
xlims=[logTicks(1), logTicks(end)+1];
nfreqs=numel(freqs);


figure, hold on
if include_og
    if normByArea
        plot(freqs,normalizeByArea(mean(og_ms,1),freqs),'DisplayName','og');
    else
        plot(freqs,mean(og_ms,1),'DisplayName','og');
    end
end
for cii=1:numel(show_conditions)
    cond_label=show_conditions{cii};
    switch cond_label
        % order corresponds to index in warped_ms
        case 'irregular'
            cc=1;
        case 'regular'
            cc=2;
    end
    for rii=1:numel(show_rules)
        show_rule=show_rules(rii);
        for fii=1:numel(show_fcenters)
            show_center=show_fcenters{fii};

            %TODO: change this so display name is a variable and plot
            %command runs outside of if else block
            ms_line=mean(warped_ms(show_rule,cc).(show_center),1);
            if normByArea
                ms_line=normalizeByArea(ms_line,freqs);
            end
            if numel(show_fcenters)==1 && numel(show_rules)==1
                line_label=cond_label;
            else
                if numel(show_fcenters)>1 && numel(show_rules)==1
                    line_label=[cond_label '-' show_center];
                elseif numel(show_fcenters)==1 && numel(show_rules)>1
                    line_label=sprintf('%s - rule %d',cond_label,show_rule);
                    % line_label=[cond_label '-' show_rule];
                end
            end
            h=plot(freqs,ms_line,'DisplayName',line_label);
            % if normByArea
            %     h=plot(freqs,normalizeByArea(mean(warped_ms(show_rule,cc).(show_center),1),freqs),'DisplayName',cond_label);
            % else
            %     h=plot(freqs,mean(warped_ms(show_rule,cc).(show_center),1),'DisplayName',cond_label);
            % end
        end
    end
end

if normByArea
    norm_str='area normalized';
else
    norm_str='unnormalized';
end
if numel(show_rules)==1 && numel(show_fcenters)==1
    title_str=sprintf('Rule %d - %s - %s',show_rule,show_center,norm_str);
elseif numel(show_conditions)==1
    if numel(show_rules)==1 && numel(show_fcenters)>1
        title_str=sprintf('%s - Rule %d vs frequency center -%s', ...
            show_conditions{:},show_rules,norm_str);
    elseif numel(show_rules)>1 && numel(show_fcenters)==1
        title_str=sprintf('%s - %s vs rules - %s', ...
            show_conditions{:},show_fcenters{:},norm_str);

    end

end

title(title_str);
xlabel('frequencies')
legend()
set(gca, 'Xscale','log','XTick',logTicks,'XLim',xlims)
hold off

   