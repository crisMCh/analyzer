function [nps, f] = funcCalcDigitalNps(I, n, px, use_window, average_stack)
%% Function summary
% Compute the NPS in a given ROI (I) 

% Function Inputs:

% Function Returns:
%   

% Updates:
% 

%% Check input arguments
if nargin<3 || isempty(px), px=1; end
if nargin<4 || isempty(use_window), use_window=0; end
if nargin<5 || isempty(average_stack), average_stack=0; end

%% Parameter defintions 
stack_size=size(I,n+1);

size_I=size(I);
if any(diff(size_I(1:n)))
    error('ROI must be symmetric.');
end
roi_size=size_I(1);

% Cartesian coordinates
x=linspace(-roi_size/2,roi_size/2,roi_size);
x=repmat(x',[1 ones(1,n-1)*roi_size]);

% frequency vector
f=linspace(-0.5,0.5,roi_size)/px;

% Radial coordinates
r2=0; for p=1:n, r2=r2+shiftdim(x.^2,p-1); end, r=sqrt(r2);

%% Preprocessing

% Hann window to avoid spectral leakage
if use_window
    h=0.5*(1+cos(pi*r/(roi_size/2)));
    h(r>roi_size/2)=0;
    h=repmat(h,[ones(1,n) stack_size]);
else h=1;
end

% Detrending by subtracting the mean of each ROI
S=I; for p=1:n, S=repmat(mean(S,p), ((1:n+1)==p)*(roi_size - 1) + 1); end
%F=(I-S).*h; %Detrending --> estefade nemishe
F=(I).*h; %activate the detrending 

%% NPS computation

% 2D-FFT
%for p = 1:n, F = fftshift(fft(F,[],p),p); end %Original fft version
F = fftshift(fft2(F));

% Compute the cartesian NPS
nps=abs(F).^2/...
    roi_size^n*px^n./... NPS in units of px^n
    (sum(h(:).^2)/length(h(:))); % the normalization with h is according to Gang 2010

% Averaging the NPS over the ROIs (assuming ergodicity)
if average_stack, nps=mean(nps,n+1); end  

