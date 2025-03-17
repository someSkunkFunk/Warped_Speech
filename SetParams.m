scrcenter = outRect(3:4)./2;
height = outRect(4);

% Text coordinates
text_xQ = outRect(3)/2-500;
text_xA = outRect(3)/2-250;

text_yQ = outRect(4)/2-250;
text_yA1 = outRect(4)/2-150;
text_yA2 = outRect(4)/2-50;
text_yA3 = outRect(4)/2+50;
text_yA4 = outRect(4)/2+150;
text_yP = outRect(4)/2+250;

fixcolor = [100 0 0];
cRed0 = [150 0 0];
cGreen0 = [0 150 0];
cWhite0 = [180 180 180];
cBg0 = [70 70 70];

fixsize = 10;
dur = 180;
Fs = 48000;
aAmp = 0.1;
deviceID = audioDeviceID('front');
blobsigma = outRect(4)/10;
aDelay = 0.013;
vDelay = 0;
trg = 0.0023;

Lmargin = 400;
firstL = 300;
H = 50;