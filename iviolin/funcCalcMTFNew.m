function [xaxis,MTF,MTFarea] = funcCalcMTFNew(myimg,ActContMask,PixSpc,ROIname,PatientNumber,DEBUG_PLOT_LEVEL,ENV_TYPE,HOST_TYPE)
%% Function summary
% Date: 11.09.2020  
% Compute the MTF of the image/ROI

% Function Inputs:

% Function Returns:
%   

% Updates:
% 14.03.21: Corrections in the MTF with "ImageInterPFactor" 


%% Important steps of this script

%0. Set global parameters
%1. Preprocessing (only used for improving edge detection)
%2. Edge detection using active contours
%3. Gradient estimation along edges
%4. Line profiles estimation perpendicular to the detected edge
%5. Bump removal in ESF curves
%6. Alignment and averaging of ESF and LSF curves
%7. MTF using average LSF

%% (0) Load image and set global parameters
ImShowSizeX=600;ImShowSizeY=600; %Displayed image size using imshow

%% PLOT - Show selected ROI - original before inerpolation
if DEBUG_PLOT_LEVEL >= 1
    [ySize,xSize]=size(myimg);
    h=figure;hold on;
        %imshow(double(myimg)/max(max(double(myimg))));colormap(gray);truesize([ImShowSizeX ImShowSizeY])
        imagesc(flipud(myimg));colormap(gray);
        if ENV_TYPE == 0, truesize([ImShowSizeX ImShowSizeY]), end
        xlim([1 xSize])
        ylim([1 ySize])
        set(gca,'visible','off') %Hide axes
        %Export image as EPS and JPG and PNG
        imgPath = strcat('Images\');
        imgName = strcat('Pat-',int2str(PatientNumber),'-MTF', '-ROI-Original');
        imgPathFnameFull = strcat(imgPath,imgName);
        if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
        saveas(h,strcat(imgPathFnameFull,'.png'));
        print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end

%% (1) Preprocessing and parameter definition (only for edge detection)

% Make a copy of the original, unprocessed image
%myimgorig=myimg;
myimg=double(myimg);
ImageInterPFactor=2; %Interpolation factor used to linearly interpolate the 2D image matrix
myimg = interp2(myimg,ImageInterPFactor);
myimg=int16(myimg);
myimgorig=myimg;
PixSpc = PixSpc / ImageInterPFactor;

%% PLOT - Show selected ROI after interpolation
%Plot edge in image
if DEBUG_PLOT_LEVEL >= 2
    [ySize,xSize]=size(myimg);
    h = figure;hold on;
        %imshow(double(myimg)/max(max(double(myimg))));colormap(gray);truesize([ImShowSizeX ImShowSizeY])
        imagesc(flipud(myimg));colormap(gray);
        if ENV_TYPE == 0, truesize([ImShowSizeX ImShowSizeY]), end
        xlim([1 xSize])
        ylim([1 ySize])
        set(gca,'visible','off') %Hide axes

        %Export image as EPS and JPG and PNG
        imgPath = strcat('Images\');
        imgName = strcat('Pat-',int2str(PatientNumber),'-MTF', '-ROI-Interp');
        imgPathFnameFull = strcat(imgPath,imgName);
        if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
        saveas(h,strcat(imgPathFnameFull,'.png'));
        print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));

end    
    
%% Preprocessing to enhance the quality of the edge detection
% No used for the lines profiles, only for the edge detection step
myimg = imadjust(myimg);
myimg = imadjust(myimg);
if ENV_TYPE == 0 %MATLAB
  myimg = imgaussfilt(myimg,2);
elseif ENV_TYPE == 1 %OCTAVE
  myimg = imsmooth(myimg,"Gaussian"); 
end
myimg=imsharpen(myimg);

%figure;imagesc(myimg);colormap(gray);truesize([ImShowSizeX ImShowSizeY])
%% PLOT - Show selected ROI after interpolation and image enhancement
if DEBUG_PLOT_LEVEL >= 2
    [ySize,xSize]=size(myimg);
    h = figure;hold on;
        %imshow(double(myimg)/max(max(double(myimg))));colormap(gray);truesize([ImShowSizeX ImShowSizeY])
        imagesc(flipud(myimg));colormap(gray);
        if ENV_TYPE == 0, truesize([ImShowSizeX ImShowSizeY]), end
        xlim([1 xSize])
        ylim([1 ySize])
        set(gca,'visible','off') %Hide axes

        %Export image as EPS and JPG and PNG
        imgPath = strcat('Images\');
        imgName = strcat('Pat-',int2str(PatientNumber),'-MTF', '-ROI-Interp-ImgEnhance');
        imgPathFnameFull = strcat(imgPath,imgName);
        if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
        saveas(h,strcat(imgPathFnameFull,'.png'));
        print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end    
    
%% (2) Edge detection using active contours

% (2.0) Parameter settings for the active contours method

% Mask definition for active contours and image resize (50%)
m = zeros(size(myimg,1),size(myimg,2));          %create initial mask
maskX = ImageInterPFactor^2*ActContMask.x;
maskY = ImageInterPFactor^2*ActContMask.y;
WinWidth = ImageInterPFactor*2; 
m(maskY-WinWidth:maskY+WinWidth,maskX-WinWidth:maskX+WinWidth)=1; 

[imgysize,imgxsize] = size(myimg);

% (2.1) Active contours
if DEBUG_PLOT_LEVEL <= 0
    seg = funcRegionSeg(myimg, m, 1000,0.2,0); %-- Run segmentation (alpha=0.2 (DEFAULT), PLOT DISABLED) 
elseif DEBUG_PLOT_LEVEL > 0
    seg = funcRegionSeg(myimg, m, 1000); %-- Run segmentation
end


if DEBUG_PLOT_LEVEL >= 2
    figure
    subplot(2,2,1); imshow(myimg); title('Input Image');
    subplot(2,2,2); imshow(m); title('Initialization region');
    subplot(2,2,4); imshow(seg); title('Global Region-Based Segmentation');
end

myimgseg = seg;

% (2.2) Generate edge matrix (edge pixels == 1, background == 0)
if ENV_TYPE == 0 %MATLAB
  myimgedge = edge(myimgseg); %Edge matrix %MATLAB
elseif ENV_TYPE == 1 %OCTAVE
  %myimgedge = edge(double(myimgseg),"Sobel"); %Edge matrix %OCTAVE %16.05.21: Sobel does not detect the edges properly in its current implementation
  myimgedge = edge(double(myimgseg),"Kirsch"); %Edge matrix %OCTAVE 
end

%Plot edge matrix
%figure; imshow((myimgedge));title('Detected edge');truesize([ImShowSizeX ImShowSizeY])
%% PLOT - Show detected edge in ROI
if DEBUG_PLOT_LEVEL >= 1
    [ySize,xSize]=size(myimgedge);
    h = figure;hold on;
        %imshow(double(myimg)/max(max(double(myimg))));colormap(gray);truesize([ImShowSizeX ImShowSizeY])
        imagesc(flipud(myimgedge));colormap(gray);
        if ENV_TYPE == 0, truesize([ImShowSizeX ImShowSizeY]), end
        xlim([1 xSize])
        ylim([1 ySize])
        set(gca,'visible','off') %Hide axes

        %Export image as EPS and JPG and PNG
        imgPath = strcat('Images\');
        imgName = strcat('Pat-',int2str(PatientNumber),'-MTF', '-ROI-Edge');
        imgPathFnameFull = strcat(imgPath,imgName);
        if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
        saveas(h,strcat(imgPathFnameFull,'.png'));
        print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end    

[yEdge,xEdge]=find(myimgedge==1);
    
%Plot edge in image
if DEBUG_PLOT_LEVEL >= 2
    h = figure;
    %imshow(myimg);colormap(gray);truesize([ImShowSizeX ImShowSizeY])
    imagesc(myimgorig);colormap(gray);
    if ENV_TYPE == 0, truesize([ImShowSizeX ImShowSizeY]), end
    hold on;
    %imagesc(myimg);colormap(gray)
    for k=1:1:length(xEdge) %har pixel ro hesab mikone, age masalan k:1:2 bashe yeki darmion hesab mikone
    rectangle('Position',[xEdge(k),yEdge(k),1,1],...
              'LineWidth',2,'EdgeColor','g')
    end
end

%% PLOT - Show ROI and overlayed edge pixels as rectangles
if DEBUG_PLOT_LEVEL >= 1
h = figure;hold on;
    %imshow(myimg);colormap(gray);truesize([ImShowSizeX ImShowSizeY])
    imagesc((myimg));colormap(gray);
    if ENV_TYPE == 0, truesize([ImShowSizeX ImShowSizeY]), end
    hold on;
    %imagesc(myimg);colormap(gray)
    for k=1:1:length(xEdge) %har pixel ro hesab mikone, age masalan k:1:2 bashe yeki darmion hesab mikone
        rectangle('Position',[xEdge(k),yEdge(k),1,1],...
            'LineWidth',2,'EdgeColor','g')
    end
    xlim([1 xSize])
    ylim([1 ySize])
    set(gca,'View',[-0 -90]) %Rotate the figure
    set(gca,'visible','off') %Hide axes

    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-MTF', '-ROI-Edge-Overlay');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end

%% (3) Gradient computation
%This computes the gradient magnitues and direction (angle) for each(!)
%pixel in the image, i.e. for the edge and for the background
[Gmag,Gdir]=imgradient(myimg);

edgemag=zeros(length(yEdge),1);%Create empty vector for edge gradient magnitudes
edgedir=zeros(length(yEdge),1);%Create empty vector for edge gradient directions
for k=1:length(yEdge) %Copy the gradient information for the edge pixels (xEdge,yEdge)
    edgemag(k,1) = Gmag(yEdge(k),xEdge(k));
    edgedir(k,1) = Gdir(yEdge(k),xEdge(k));
end


%% 17.01.2020: Plot gradient directions along one edge 
%!!Use Carefully! Always %check manually if this works!)
%xyEdgePlot = funcPlotGradDirAlongEdge(xEdge,yEdge,Gdir,myimgorig);
%[xyEdgePlot,GdirPlotStd] = funcPlotGradDirAlongEdge(xEdge,yEdge,Gdir,myimgorig,DEBUG_PLOT_LEVEL); %02.03.21: Also return the standard deviations of the angle for each edge pixel

 
%% (4) Line profile estimation based on gradient directions
% From here on, use the original, unfilterd image again
myimg=myimgorig; 

% Parameters for line profile
    %Note: Length of vector for line profile:
    %Good results with a length of 15px @ PixSpc==0.33mm -> Line length ~ 5mm
    %Use these 5mm as a rough reference
vecLenMM = 5; %Length of line profile vector in [mm] %15px @ PixSpc = 0.33mm --> length=~5mm; length of each line in pixels
vecLen = round(vecLenMM/PixSpc); %Length of line profile vector in [px] %15px @ PixSpc = 0.33mm --> length=~5mm; length of each line in pixels
LPpoints = 15; %(100) Number of points used for (interpolated) line profile

PixSpcLP = (2*vecLenMM)/LPpoints; %PixSpc*2*vecLen/LPpoints; %30.03.21: factor "2" because length of the profile is 2*vecLen

if DEBUG_PLOT_LEVEL >= 2
    h=figure;
    imagesc(myimg);colormap(gray);%truesize([ImShowSizeX ImShowSizeY]);
    hold on;title('Original (unfiltered) image used for line profile estimation');
end

%[ESF,edgePixels] = funcLineProfileNew(myimg,xyEdgePlot(:,1),xyEdgePlot(:,2),Gdir,vecLen,LPpoints,imgxsize,imgysize,PatientNumber,DEBUG_PLOT_LEVEL,ENV_TYPE); %02.03.21: Also return the ede pixels used
[ESF,edgePixels] = funcLineProfileNew(myimg,xEdge,yEdge,Gdir,vecLen,LPpoints,imgxsize,imgysize,PatientNumber,DEBUG_PLOT_LEVEL,ENV_TYPE); %02.03.21: Also return the ede pixels used
ESFbak = ESF;

%% 01.04.21 Upsample the ESF for the follwing bump removal
PixSpcLPorig = PixSpcLP;

PixSpcLP = PixSpcLPorig;
InterpBumpRemoval=4; %Interpolation factor
ESFInt = interp1(1:length(ESFbak(:,1)),ESFbak,1:1/InterpBumpRemoval:length(ESFbak(:,1)),'spline');
ESF = ESFInt; %Replace original LSF my interpolated LSF
xbak=(0:1:length(ESFbak(:,1))-1);

if DEBUG_PLOT_LEVEL >= 2
    x=(0:1:length(ESF(:,1))-1)/InterpBumpRemoval;
    figure;hold on;
        plot(xbak,ESFbak(:,1))
        plot(x,ESF(:,1))
        title('ESF, Upsamples for interpolation')
end

PixSpcLP = PixSpcLP/InterpBumpRemoval;  
   
ESF = bsxfun(@minus,ESF,mean(ESF));
LSF = diff(ESF);

%% Low pass filter the ESF  and plot the results
%[b,a] = butter(3,0.49,'low');
%[b,a] = butter(5,0.1,'low'); %!!! Only for fissure analysis 
%ESFLP=filtfilt(b,a,ESF);

ESFLP = ESF;
LSFLP=diff(ESFLP);

%% PLOT - Show ESF before and after Low Pass filtering
N = 4; 
if DEBUG_PLOT_LEVEL >= 1
    x = (0:length(ESFLP(:,1))-1)*PixSpcLP;
    h=figure('Position', [100 100 700 700]); hold on;grid on
    
    h1=plot(x,ESF(:,N),'r')
    h2=plot(x,ESFLP(:,N),'b')
    xlabel('Length [mm]')
    ylabel('Intensity [a.u.]')
    xlim([0 2*vecLenMM-1])
    set(h1, 'LineWidth', 4);
    set(h2, 'LineWidth', 4);
    set(gca, 'Box', 'off' ); 
    set(gca, 'TickDir', 'out','LineWidth',2 ,'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12); 
    legend('ESF','ESF (filtered)','location','southeast')

    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-ESF', '-LP-filt');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end

ESF = ESFLP;
LSF = LSFLP;

if DEBUG_PLOT_LEVEL >= 2
    figure;plot(ESF);title('Low pass filtered ESF');
    figure;plot(LSF);title('Low pass filtered LSF');
end

%% 26.02.2021: New bumpremoval version 

%Define the range where the edge should start / end
idxLeftLimit  = round(0.20*length(ESF(:,1)));
idxRightLimit = round(0.80*length(ESF(:,1)));

%Define the range where the LSF peak should be (with respect to the center
%of the LSF vector)
idxPeakLeftLimit  = round(0.40*length(ESF(:,1)));
idxPeakRightLimit = round(0.60*length(ESF(:,1)));

edgeCenterDistance = round(0.35*length(ESF(:,1)));

%idxLeftLimit  = 25; %02.03.2021 -> Define relative to idxCenter
%idxRightLimit = 25; %02.03.2021


idxResult=1; %Index variable for writing to esfNoBump after succesful bump removal
esfNoBump=[];
if DEBUG_PLOT_LEVEL >= 2
    figure;hold on;
end
idxValid = []; %02.03.2021: Index of valid ESF function, i.e. the one we are using.
for idxCol = 1:length(ESF(1,:))
    idxZero = funcZeroCrossing(LSF(:,idxCol)); %Zero corssing detection in LSF
    
    [val,idxCenter] = max(abs(LSF(idxPeakLeftLimit:idxPeakRightLimit,idxCol))); %Center positon of the edge hopefully corresponds to the maximum of the LSF...)
    idxCenter = idxCenter + idxPeakLeftLimit - 1;
        
    %Find first zero crossing to the left of the center position
    idxLeft = find(idxZero<idxCenter);
    if ~isempty(idxLeft)
        idxLeft = idxLeft(end);
        idxLeft = idxZero(idxLeft);
        idxLeft = idxLeft + 1; %There is a shift of 1 in the zero corssing index
    end
    
    %Find first zero crossing to the right of the center position
    idxRight = find(idxZero>idxCenter);
    if ~isempty(idxRight)
        idxRight = idxRight(1);
        idxRight = idxZero(idxRight) + 1;
    end
    
    if idxCenter - idxLeft <  edgeCenterDistance & idxRight - idxCenter <  edgeCenterDistance
        tmp = ESF(:,idxCol);
        tmp(1:idxLeft) = tmp(idxLeft);
        tmp(idxRight:end) = tmp(idxRight);
        esfNoBump(:,idxResult) = tmp;
        tmp = [];
        
        idxValid(idxResult) = idxCol; %02.03.21
        
        idxResult = idxResult + 1;
        
        if DEBUG_PLOT_LEVEL >= 2
            plot(ESF(:,idxCol));
            plot(idxLeft,ESF(idxLeft,idxCol),'rd');
            plot(idxRight,ESF(idxRight,idxCol),'gd');
            title('Detected bump positions (considered as the end of the edge)');
        end
    end       
end

if DEBUG_PLOT_LEVEL >= 2
    figure;plot(esfNoBump)
    figure;plot(diff(esfNoBump))
end

ESF = esfNoBump;
LSF = diff(ESF);


%% 01.04.21: Downsample the ESF (was upsampled for better bump removal)
ESFd = downsample(ESF,InterpBumpRemoval);
PixSpcLP = PixSpcLP*InterpBumpRemoval; %Reset pixel spacing to original value

if DEBUG_PLOT_LEVEL >= 2
    xbak=(0:1:length(ESFbak(:,1))-1);
    x=(0:1:length(ESFd(:,1))-1);
    figure;hold on;
        plot(xbak,ESFbak(:,1)-mean(ESFbak(:,1)))
        plot(x,ESFd(:,1))
end
    
ESF = ESFd;
LSF = diff(ESF);

%% 31.03.2021 %Extend the ESF to the left and right to generate more data points
%Fill beginning and end with zeros
ESFT=ESF;
ESFTlen = length(ESFT(:,1));
ESFTfill = zeros(ESFTlen,length(ESFT(1,:)));
ESFT = [ESFTfill;ESFT];
ESFT = [ESFT;ESFTfill];
for k=1:length(ESFT(1,:))
    ESFT(1:ESFTlen,k) = ESFT(ESFTlen+1,k)*ones(ESFTlen,1);
    ESFT(2*ESFTlen+1:end,k) = ESFT(2*ESFTlen,k)*ones(ESFTlen,1);
end

ESF=ESFT; %31.03.21
LSF = diff(ESF);%31.03.21

%% PLOT - ESF before anf after bump removal without extension
N = 3; 
if DEBUG_PLOT_LEVEL >= 1
    x = (0:length(ESFbak(:,1))-1)*PixSpcLP;
    h=figure('Position', [100 100 700 700]);hold on;grid on
    
    h1=plot(x,ESFbak(:,N)-mean(ESFbak(:,N)));
    h2=plot(x,ESFd(:,N)-mean(ESFd(:,N)));
    %h2=plot(x,ESFLP(:,N))
    %imshow(myimg);colormap(gray);truesize([ImShowSizeX ImShowSizeY])
    xlabel('Length [mm]')
    ylabel('Intensity [a.u.]')
    %xlim([0 2*vecLenMM-1])
    xlim([0 x(end)])
    set(h1, 'LineWidth', 4);
    set(h2, 'LineWidth', 4);
    set(gca, 'Box', 'off' ); 
    set(gca, 'TickDir', 'out','LineWidth',2 ,'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12); 
    legend('ESF','ESF (edge only)','location','southeast')

    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-ESF', '-Bump-Removal-Comparison');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end

%% PLOT - ESF before anf after bump removal without extension
N = 3; 
if DEBUG_PLOT_LEVEL >= 1
    x = (0:length(ESFbak(:,1))-1)*PixSpcLP;
    h=figure('Position', [100 100 700 700]);hold on;grid on
    
    h1=plot(x,ESFbak(:,N)-mean(ESFbak(:,N)));
    h2=plot(x,ESFd(:,N)-mean(ESFd(:,N)));
    %h2=plot(x,ESFLP(:,N))
    %imshow(myimg);colormap(gray);truesize([ImShowSizeX ImShowSizeY])
    xlabel('Length [mm]')
    ylabel('Intensity [a.u.]')
    %xlim([0 2*vecLenMM-1])
    xlim([0 x(end)])
    set(h1, 'LineWidth', 4);
    set(h2, 'LineWidth', 4);
    set(gca, 'Box', 'off' ); 
    set(gca, 'TickDir', 'out','LineWidth',2 ,'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12); 
    legend('ESF','ESF (edge only)','location','southeast')

    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-ESF', '-Bump-Removal-Comparison');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end



%% PLOT - LSF after bump removal without extensions
if DEBUG_PLOT_LEVEL >= 1
    x = (0:length(diff(ESFd(:,1)))-1)*PixSpcLP;
    h=figure('Position', [100 100 700 700]);hold on;grid on
    
    h1=plot(x,diff(ESFd(:,N)));
    %h2=plot(x,ESFLP(:,N))
    %imshow(myimg);colormap(gray);truesize([ImShowSizeX ImShowSizeY])
    xlabel('Length [mm]')
    ylabel('Intensity [a.u.]')
    %xlim([0 2*vecLenMM-1])
    xlim([0 x(end)])
    set(h1, 'LineWidth', 4);
    %set(h2, 'LineWidth', 4);
    set(gca, 'Box', 'off' ); 
    set(gca, 'TickDir', 'out','LineWidth',2 ,'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12); 
    %legend('ESF','ESF (filtered)','location','southeast')

    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-LSF', '-NoExtension');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end



%% PLOT - LSF 
if DEBUG_PLOT_LEVEL >= 1
    x = (0:length(LSF(:,1))-1)*PixSpcLP;
    h=figure('Position', [100 100 700 700]); hold on;grid on
    h1=plot(x,LSF(:,N));
    %h2=plot(x,ESFLP(:,N))
    %imshow(myimg);colormap(gray);truesize([ImShowSizeX ImShowSizeY])
    xlabel('Length [mm]')
    ylabel('Intensity [a.u.]')
    %xlim([0 2*vecLenMM-1])
    xlim([0 x(end)])
    set(h1, 'LineWidth', 4);
    %set(h2, 'LineWidth', 4);
    set(gca, 'Box', 'off' ); 
    set(gca, 'TickDir', 'out','LineWidth',2 ,'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12); 

    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-LSF');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));

end

%Interpolation if required
InterpFactor=1; %Interpolation factor
LSFInt = interp1(1:length(LSF(:,1)),LSF,1:1/InterpFactor:length(LSF(:,1)),'spline');
LSF = LSFInt; %Replace original LSF my interpolated LSF

%% (7) MTF
MTF = abs(fft(LSF));

%Normalize MTF
MTF = bsxfun(@rdivide,MTF,MTF(2,:));

%Plot with correct axis label
xaxis = (0:1:length(MTF(:,1))-1)/(length(MTF(:,1))-1)*InterpFactor/PixSpcLP; %14.03.2021: Consinder PixSpcLP instead of PixSpcfrom interpolating the line profile
%figure;h1=plot(xaxis(2:end),MTF(2:end)/MTF(2));title('MTF computed from the mean LSF');
   
%% Show some debugging information about pixel spacing and sampling frequency (sampling of the line profile)    
disp('*****************************************************************')
disp('MTF frequency-axis based on the line profile sampling frequency: ')
disp(['Length of the line profile [mm] = ',num2str(vecLenMM)])
disp(['Number of pixels = ',num2str(LPpoints)])
disp(['Sampling frequency [Pixel/mm] = ',num2str(LPpoints/vecLenMM)])
disp(['Sampled pixel spacing [mm] = ',num2str(1/(LPpoints/vecLenMM))])
disp('*****************************************************************')
disp(' ')

%% PLOT - MTF - ALL CURVES
if DEBUG_PLOT_LEVEL >= 1
    h=figure('Position', [100 100 700 700]); hold on
    
    grid on
    for k=1:length(MTF(1,:))
        h1=plot(xaxis(2:end),MTF(2:end,k));
        set(h1, 'LineWidth', 4);
    end
    xlabel('Spatial frequency [1/mm]');
    ylabel('MTF(f) (normalized)')
    xlim([xaxis(2) 1/PixSpcLP/2])
    ylim([0 1])
    set(gca, 'Box', 'off' ); 
    set(gca, 'TickDir', 'out','LineWidth',2, 'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12);

    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-MTF-All');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end

%% PLOT - MTF - MEAN CURVE
if DEBUG_PLOT_LEVEL >= 1    
    h=figure('Position', [100 100 700 700]); hold on
    
    grid on
    h1=plot(xaxis(2:end),mean(MTF(2:end,:)'));
    xlabel('Spatial frequency [1/mm]');
    ylabel('MTF(f) (normalized)')
    xlim([xaxis(2) 1/PixSpcLP/2])
    ylim([0 1])
    set(h1, 'LineWidth', 4);
    set(gca, 'Box', 'off' ); 
    set(gca, 'TickDir', 'out','LineWidth',2, 'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12);

    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-MTF-Mean');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end
    
    
%% Compute the area under the MTF (half the frequency range, i.e. up to the Nyquist frequency)
ElemNum = floor(length(MTF(2:end,1))/2) - 1;
MTFarea=trapz(xaxis(2:ElemNum),MTF(2:ElemNum,:));   
%disp(strcat('Mean MTF area (under curve):', float2str(mean(MTFarea))))
sprintf('Mean MTF area (under curve): %.3f',mean(MTFarea))


%% PLOT - EDGE USED FOR ANALYSIS OF GRADIENT ANGLES AND MTF (ALONG THIS EDGE)
if DEBUG_PLOT_LEVEL >= 1
    h=figure('Position', [100 100 700 700]); hold on
    grid on
    h1=imagesc(myimgorig);colormap(gray);%truesize([ImShowSizeX ImShowSizeY]);
    for k=1:1:length(edgePixels(:,1))-1 %har pixel ro hesab mikone, age masalan k:1:2 bashe yeki darmion hesab mikone
    rectangle('Position',[edgePixels(k,1),edgePixels(k,2),1,1],...
              'LineWidth',2,'EdgeColor','g')
    end
    xlim([1 xSize])
    ylim([1 ySize])
    set(gca,'View',[-0 -90]) %Rotate the figure
    set(gca,'visible','off') %Hide axes

    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-EdgeAnalysis-EdgePixel');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end
    
%% PLOT - MTF AREA ALONG THE EDGE
if DEBUG_PLOT_LEVEL >= 1
    h=figure('Position', [100 100 700 700]); hold on
    grid on
    h1=plot(MTFarea,'d');
    xlabel('Edge pixel')
    ylabel('Area under MTF curve')
    %xlim([xaxis(2) 4])
    ylim([0 max(max(MTFarea))])
    set(h1, 'LineWidth', 4);
    set(gca, 'Box', 'off' ); 
    set(gca, 'TickDir', 'out','LineWidth',2, 'TickLength',[0.02 0.08]);
    set(gca,'FontSize',12);

    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-EdgeAnalysis-MTFArea');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
end

%% PLOT - Standard deviation of the gradient angles around edge pixels
% if DEBUG_PLOT_LEVEL >= 1
%     h=figure('Position', [100 100 700 700]); hold on
%     grid on
%     h1=plot(GdirPlotStd(idxValid),'d');
%     xlabel('Edge pixel')
%     ylabel('Standard deviation')
%     %xlim([xaxis(2) 4])
%     ylim([0 max(max(GdirPlotStd(idxValid)))])
%     set(h1, 'LineWidth', 4);
%     set(gca, 'Box', 'off' ); 
%     set(gca, 'TickDir', 'out','LineWidth',2, 'TickLength',[0.02 0.08]);
%     set(gca,'FontSize',12);
% 
%     %Export image as EPS and JPG and PNG
%     imgPath = strcat('Images\');
%     imgName = strcat('Pat-',int2str(PatientNumber),'-EdgeAnalyis-StdDevGradientAngles');
%     imgPathFnameFull = strcat(imgPath,imgName);
%     if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
%     saveas(h,strcat(imgPathFnameFull,'.png'));
%     print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
% end    

%% Save results in a txt-file

% ##if HOST_TYPE == 0
% ##  txtPath = strcat('Results\');%strcat(ImagePathInfo.PathResults);
% ##  txtName = strcat('Pat-',int2str(PatientNumber),'-Results-MTF') %strcat(ImagePathInfo.FName,'-MTFarea','-',ROIname);
% ##  txtPathFnameFull = strcat(txtPath,txtName);
% ##
% ##  fileID = fopen(strcat(txtPathFnameFull,'.txt'),'w');
% ##  fprintf(fileID,'-------------------------------------------------------\n');
% ##  if ENV_TYPE == 0 %MATLAB
% ##    fprintf(fileID,'Results timestamp: %s\n',datetime);
% ##  end
% ##  fprintf(fileID,'Results for Dataset: %s  ## image %s (Patient Number used in Matlab: %d) \n',ImagePathInfo.SubFolderName, ImagePathInfo.FName, PatientNumber);
% ##  fprintf(fileID,'Mean area under MTF curve  = %6.3f \n',mean(MTFarea));
% ##
% ##  fprintf(fileID,'\n \n \n');
% ##  fclose(fileID);
% ##end

end %End of this function
