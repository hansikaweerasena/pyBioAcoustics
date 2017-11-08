function [ out ] = moving_avg( in, windowSize, overlap)
%MOVING_AVG Summary of this function goes here
%   Detailed explanation goes here

l_in = length(in);
halfWindow = windowSize/2;

i = halfWindow;
iMax = l_in - halfWindow;

noOverlap = windowSize - overlap;

%window = (hamming(windowSize))'; % hamming window

window = (kaiser(windowSize,2))'; % kaiser window

%out = zeros(1,windowSize - overlap);
out = [];


preAvg = in(1);

while i <= iMax
    %avg = mean(in(i-halfWindow+1:i+halfWindow) .* window);
    avg = mean(in(i-halfWindow+1:i+halfWindow) .* window);
    temp = linspace(preAvg,avg,noOverlap);
    out = cat(2,out,temp);
    i = i + noOverlap;
    preAvg = avg;
end

i = i - noOverlap;

%window = hamming((l_in-i)*2);
window = kaiser((l_in-i)*2,2);

window = (window(1:l_in-i))';
avg = mean(in(i+1:l_in) .* window);
last = linspace(preAvg,avg,l_in-i);
out = cat(2,out,last);


end

