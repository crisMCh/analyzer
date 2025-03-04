#!/usr/bin/octave -qf
%BODYCONTOUR Body contour on CT images (DICOM)
%   img: the image on which the algorithm will be applied (HU converted)

pkg load image;

arg_list = argv ();
fname=arg_list{1};

id=arg_list{2};

img = dlmread(fname);

img(img<-1000)= -1000;
%dlmwrite('img_test.txt', img);

% Determine body mask
se        = strel('square', 12);
img2      = imopen(int16(img), se); % Diffuse image
%img2      = imbinarize(img2); % BW
img2      = im2bw(img2, graythresh(img2)); % BW
%dlmwrite('test3.txt', img2);
%bodymask  = imfill(img2, 'holes');
bodymask = bwfill(img2, "holes");
%dlmwrite('test2.txt', bodymask);

% Check if the bodymask contains other smaller objects and label them
[L,n] = bwlabel(bodymask);

if n>1
    % Find the largest mask
    for km=1:n
        sm(km) = sum(L(L==km))./km;
    end
    try
    	[~,lindx] = max(sm);
    catch err
        keyboard
    end
    L(L~=lindx) = 0;
end
bodymask = logical(L);

%keyboard
dlmwrite([id 'test.txt'], bodymask);
