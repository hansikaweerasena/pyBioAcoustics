function [ vectorout ] = segmentation( mvavg, signal, threshold )
%SEGMENTATION Summary of this function goes here
%   Detailed explanation goes here

l_vectorin = length(mvavg);
set = 0;
segments = 0;
%out = zeros(1,l_vectorin);

vectorout = {};
temp = [];


for i=1:l_vectorin
    if mvavg(i) >= threshold
        temp = cat(1,temp,signal(i));
        set = 0;
        %out(i) = 1;
        
    elseif set == 0
        if length(temp) > 3000
            vectorout = cat(1,vectorout, temp);
            segments = segments + 1;
        end
        temp = [];
        set = 1;
    end
end

%{
for j=1:segments
    s = sprintf('segment %d.wav',j);
    audiowrite(s,vectorout{j}, 48000);
end
%}