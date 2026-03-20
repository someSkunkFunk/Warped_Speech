% Syllable Interval Probability Density Functions
% Plots: (1) Uniform PDF [120, 500] ms and (2) Impulse at 220 ms

figure('Units','inches','Position',[1 1 4 2.5]);
hold on;

%% (1) Uniform PDF between 120 ms and 500 ms
a = 120;  % lower bound (ms)
b = 500;  % upper bound (ms)
height = 1 / (b - a);  % PDF height = 1/(b-a)

% Draw the uniform distribution as a rectangle
x_uniform = [a, a, b, b];
y_uniform = [0, height, height, 0];
plot(x_uniform, y_uniform, 'k-', 'LineWidth', 2);

%% (2) Impulse (Dirac delta approximation) at 220 ms
impulse_loc = 220;  % ms
impulse_height = 1;  % Normalized height for display
impulse_color=[255 0 0]./255;
% Draw the impulse as a vertical arrow/line with a filled arrowhead
quiver(impulse_loc, 0, 0, impulse_height, 0, ...
    'Color', impulse_color, ...
    'LineWidth', 2.5, ...
    'MaxHeadSize', 0.3);

% Add delta function label
text(impulse_loc + 8, impulse_height * 0.0033, ...
    '\delta(t - 220)', ...
    'FontSize', 9, 'Color', impulse_color, 'FontAngle', 'italic');

%% Formatting
xlabel('Target Syllable Interval (ms)', 'FontSize', 11);
ylabel('Probability Density', 'FontSize', 11);
title('Target Syllable Interval PDFs', 'FontSize', 11);

% Custom y-axis to show both distributions clearly
y_max = impulse_height * 0.005;
ylim([0, y_max]);
xlim([50, 600]);

% Add a baseline
plot([50, 600], [0, 0], 'k-', 'LineWidth', 0.5);

% Legend
legend({'Uniform [120, 500] ms', 'Impulse at 220 ms'}, ...
    'Location', 'northeast', 'FontSize', 11);

% Mark key x locations
ax = gca;
% ax.XTick = [120, 220, 500];
ax.FontSize = 11;
grid on;
ax.GridAlpha = 0.3;

% Annotate uniform PDF height
text((a+b)/2, height + 0.0003, ...
    sprintf('h = 1/(b-a) = %.4f', height), ...
    'HorizontalAlignment', 'center', 'FontSize', 9, 'Color', [0.3 0.3 0.3]);

hold off;