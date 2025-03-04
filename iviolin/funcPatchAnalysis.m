function b = funcPatchAnalysis(IMG,PatchSizeX,PatchSizeY,DEBUG_PLOT_LEVEL,ENV_TYPE,PatientNumber)
%% Function summary
% Divide the ROI into small patches and analyse various features within
% these patches to differentiate between homogenous and inhomogenous image
% regions using k-means clustering
%
% Function Inputs:
% 
% Function Returns:
%   

% Updates:

%% Compute parameters for each patch
[ySize,xSize]=size(IMG);

PatchStd=[];
PatchStdGmag=[];
PatchStdGdir=[];
Gmag=[];
Gdir=[];
%PatchSize=3; %OLD
for n=1:1:ySize-PatchSizeY+1
    for m=1:1:xSize-PatchSizeX+1
        PatchTmp = IMG(n:n+PatchSizeY-1,m:m+PatchSizeX-1);
        
        %Standard deviation
        PatchStd(n,m)=std2(PatchTmp);
        
        %Gradient analysis
        if ENV_TYPE == 0 %MATLAB
          [Gmag,Gdir]=imgradient(PatchTmp);
        elseif ENV_TYPE == 1 %OCTAVE
          PatchTmpGray = IMG-min(IMG(:));
          PatchTmpGray=PatchTmpGray./max(PatchTmpGray(:));
          [Gmag,Gdir]=imgradient(double(PatchTmpGray));
        end
        PatchStdGmag(n,m)=std2(Gmag);
        PatchStdGdir(n,m)=std2(Gdir);
        
        %Other statistics
        PatchEntropy(n,m)=entropy(PatchTmp);
        PatchSkewness(n,m)=skewness(PatchTmp(:));
        PatchKurtosis(n,m)=kurtosis(PatchTmp(:));
        
        %Texture analysis
        PatchZigZag=funcZigZag(PatchTmp); %ZigZag transform of patch (1D)
        PatchZigZagTranspose=funcZigZag(transpose(PatchTmp)); %ZigZag transform of transposed patch (1D)
        
    end
end

%% Plot some images
if DEBUG_PLOT_LEVEL >= 2
    figure;imagesc(PatchStd);colormap(gray);title('Standard deviation of patches')
    figure;imagesc(PatchStdGmag);colormap(gray);title('Standard deviation of gradient magnitude of patches')
    figure;imagesc(PatchStdGdir);colormap(gray);title('Standard deviation of gradient directions of patches')
    figure;imagesc(PatchEntropy);colormap(gray);title('Entropy of patches')
    figure;imagesc(PatchSkewness);colormap(gray);title('Skewness of patches')
    figure;imagesc(PatchKurtosis);colormap(gray);title('Kurtosis of patches')
    
    %% PLOT - NPS ROI
    h=figure('Position', [100 100 700 700]); hold on
    h1=imagesc(IMG);colormap(gray);
    set(gca,'visible','off') %Hide axes
    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-NPS-Selected-ROI');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
    
    %% PLOT - Standard deviation of patches
    h=figure('Position', [100 100 700 700]); hold on
    h1=imagesc(PatchStd);colormap(gray);
    set(gca,'visible','off') %Hide axes
    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-NPS-Selected-ROI-StdDev');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));

    %% PLOT - Standard deviation of gradient magnitude of  patches
    h=figure('Position', [100 100 700 700]); hold on
    h1=imagesc(PatchStdGmag);colormap(gray);
    set(gca,'visible','off') %Hide axes
    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-NPS-Selected-ROI-StdDevGradMag');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
    
    %% PLOT - Standard deviation of gradient magnitude of  patches
    h=figure('Position', [100 100 700 700]); hold on
    h1=imagesc(PatchStdGdir);colormap(gray);
    set(gca,'visible','off') %Hide axes
    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-NPS-Selected-ROI-StdDevGradDir');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
    
end

%% Analyze patch data

%PatchData = [PatchStd(:),PatchStdGmag(:),PatchStdGdir(:),PatchEntropy(:)];
PatchData = [PatchStd(:),...
    PatchStdGmag(:),...
    PatchStdGdir(:),...
    PatchEntropy(:),... 
    PatchSkewness(:),... 
    PatchKurtosis(:)
    ];
    
xSizePatchData = xSize - PatchSizeX + 1;
ySizePatchData = ySize - PatchSizeY + 1;
%k-means Default
[idx,C] = kmeans(PatchData(1:1:ySizePatchData*xSizePatchData,:),2);


%% Perform the k-Means clustering

%k-Means - Detailed
if ENV_TYPE == 0 %MATLAB
opts = statset('Display','final');
[idx,C] = kmeans(PatchData(1:1:ySizePatchData*xSizePatchData,:),2,'Distance','cityblock',...
    'Replicates',5,'Options',opts);
elseif ENV_TYPE == 1 %OCTAVE
StartMat = [1 ,1, 1, 1, 1, 1];
StartMat = [StartMat; 2*StartMat];
%[idx,C] = kmeans(PatchData(1:1:ySizePatchData*xSizePatchData,:),2,'Distance','cityblock',...
%    'Replicates',5);
[idx,C] = kmeans(PatchData(1:1:ySizePatchData*xSizePatchData,:),2,'Distance','cityblock',...
   'Replicates',1, 'start', StartMat);
end

b=reshape(idx,[ySizePatchData,xSizePatchData]);
b=-1.*(b-2);%Invert labels (instead of [1,2] make it [1,0]
if DEBUG_PLOT_LEVEL >= 1
    figure;imagesc(b);colormap(gray)
end

%% Plot some figures 
%!! SAVE THIS FIGURE
if DEBUG_PLOT_LEVEL >= 1
    h=figure;
    subplot(1,3,1)
        h1=imagesc(flipud(IMG));colormap(gray);
        title('ROI for structural analysis')
        set(gca,'XTick',[], 'YTick', []) %Hide axis labels
    subplot(1,3,2)
        h1=imagesc(flipud(IMG(1:ySizePatchData,1:xSizePatchData)));colormap(gray);
        %set(h, 'AlphaData', flipud(-1000*(b-1)));
        set(h1, 'AlphaData', flipud(1000*(b)));
        title('Class 1')
        set(gca,'XTick',[], 'YTick', []) %Hide axis labels
    subplot(1,3,3)
        h1=imagesc(flipud(IMG(1:ySizePatchData,1:xSizePatchData)));colormap(gray);
        set(h1, 'AlphaData', flipud(-1000*(b-1)));
        %set(h, 'AlphaData', flipud(1000*(b)));  
        title('Class 2')
        set(gca,'XTick',[], 'YTick', []) %Hide axis labels
    if ENV_TYPE == 0, truesize([300 300]), end
    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-NPS-Selected-ROI-Classes');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
    
    %% PLOT - NPS - Class 1
    h=figure('Position', [100 100 700 700]); hold on
    h1=imagesc((IMG(1:ySizePatchData,1:xSizePatchData)));colormap(gray);
    set(h1, 'AlphaData', (1000*(b)));
    %ax1=gca;
    %set(ax1,'YColor','r','YAxisLocation','right');
    set(gca,'XTick',[], 'YTick', []) %Hide axis labels
    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-NPS-Selected-ROI-Class1');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
    
    %% PLOT - NPS - Class 2
    h=figure('Position', [100 100 700 700]); hold on
    h1=imagesc((IMG(1:ySizePatchData,1:xSizePatchData)));colormap(gray);
    set(h1, 'AlphaData', (-1000*(b-1)));
    %ax1=gca;
    %set(ax1,'YColor','r','YAxisLocation','right');
    set(gca,'XTick',[], 'YTick', []) %Hide axis labels
    %Export image as EPS and JPG and PNG
    imgPath = strcat('Images\');
    imgName = strcat('Pat-',int2str(PatientNumber),'-NPS-Selected-ROI-Class2');
    imgPathFnameFull = strcat(imgPath,imgName);
    if ENV_TYPE == 0, hgexport(h,strcat(imgPathFnameFull,'.eps')), end
    saveas(h,strcat(imgPathFnameFull,'.png'));
    print(h,'-djpeg',strcat(imgPathFnameFull,'.jpg'));
    
    
end


%% MANUAL SELECTION: Ask user to select the backgroudn region (only if display of fiugures is enable)
b_bak = b;
%White == 1, Black == 0
if DEBUG_PLOT_LEVEL > 5 %THIS IS NOT CALLED ANYMORE
    x = input('Which figure (Class 1 or Class 2) represents the backround (white)?');
    if x == 1
        b = -1*(b-1);
    end
end

%% AUTOMATIC SELECTION: Chose background based on variance
if mean2(PatchStd(b_bak==0)) < mean2(PatchStd(b_bak==1))
    b_bak = -1*(b_bak-1);
end

if DEBUG_PLOT_LEVEL > 5 %THIS IS NOT CALLED ANYMORE
    if isequal(b,b_bak)
        input('Manual selection == automatic selection. Press ENTER to continue...');
    else
        input('WARNING: Manual selection NOTEQUAL automatic selection. Press ENTER to continue...');
    end
end


b = b_bak;

return %End of this function