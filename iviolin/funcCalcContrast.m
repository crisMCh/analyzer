function [ContrastRMS] = funcCalcContrast(IMG,ImagePathInfo,ROIname)
%% Function summary
% Date: 09.09.2020
% Compute the contrast of the image/ROI

% Definitions: 
% Method 1: RMS contrast

% Function Inputs:

% Function Returns:
%   

% Updates:



%% Additional parameter definition

%% RMS contrast
imgMean = mean2(IMG);
x = (IMG - imgMean)/imgMean;
ContrastRMS = rms(x(:));

return %End of this function