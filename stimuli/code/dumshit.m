x0=1:50;
xr=10;
y=reflect_about(x0,xr);
figure
plot(x0,'DisplayName','x0')
hold on
plot(y+5,'DisplayName','y');
legend()
hold off

function y=reflect_about(x,xr)
    y=x-2.*(x-xr);
end
